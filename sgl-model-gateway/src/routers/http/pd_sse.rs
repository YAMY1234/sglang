//! Recognize a complete SSE control event without inspecting JSON string contents.
//!
//! A reqwest byte chunk is not an SSE frame: either can span several of the
//! other. Keep only a bounded line prefix, and forward the original bytes.

#[derive(Default)]
pub(super) struct DoneEvent {
    line: [u8; 16],
    len: usize,
    after_cr: bool,
    lines_seen: bool,
    data_lines: u8,
    done_data: bool,
    completed: bool,
}

impl DoneEvent {
    pub(super) fn feed(&mut self, bytes: &[u8]) -> bool {
        for &byte in bytes {
            if self.completed {
                break;
            }
            if self.after_cr {
                self.after_cr = false;
                if byte == b'\n' {
                    continue;
                }
            }
            if byte == b'\r' || byte == b'\n' {
                self.finish_line();
                self.after_cr = byte == b'\r';
            } else {
                if self.len < self.line.len() {
                    self.line[self.len] = byte;
                }
                // One overflow marker suffices; arbitrarily long JSON lines
                // never allocate a buffer and can never equal the sentinel.
                self.len = (self.len + 1).min(self.line.len() + 1);
            }
        }
        self.completed
    }

    fn finish_line(&mut self) {
        let mut line = &self.line[..self.len.min(self.line.len())];
        if !self.lines_seen {
            line = line.strip_prefix(b"\xef\xbb\xbf").unwrap_or(line);
        }
        self.lines_seen = true;
        if line.is_empty() {
            self.completed = self.data_lines == 1 && self.done_data;
            self.data_lines = 0;
            self.done_data = false;
        } else if line == b"data" || line.starts_with(b"data:") {
            self.data_lines = self.data_lines.saturating_add(1).min(2);
            let data = line.get(5..).unwrap_or_default();
            let data = data.strip_prefix(b" ").unwrap_or(data);
            self.done_data = self.len <= self.line.len() && data == b"[DONE]";
        }
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::DoneEvent;

    #[test]
    fn literal_in_json_does_not_finish() {
        let mut parser = DoneEvent::default();
        assert!(!parser.feed(b"data: {\"text\":\"assert chunks[-1] == \\\"data: [DONE]\\\"\"}\n\n"));
        assert!(!parser.feed(b"data: {\"text\":\"events[-1] is None  # data: [DONE]\"}\n\n"));
        assert!(!parser.feed(b"data: {\"text\":\"more output\"}\n\n"));
        assert!(parser.feed(b"data: [DONE]\n\n"));
    }

    #[test]
    fn every_transport_split_recognizes_only_complete_frame() {
        let bytes = b": keepalive\r\nevent: message\r\ndata: [DONE]\r\n\r\n";
        for split in 0..bytes.len() - 1 {
            let mut parser = DoneEvent::default();
            assert!(!parser.feed(&bytes[..split]), "split {split}");
            assert!(parser.feed(&bytes[split..]), "split {split}");
        }
        let mut parser = DoneEvent::default();
        for &byte in b"data: [DONE]\n" {
            assert!(!parser.feed(&[byte]));
        }
        assert!(parser.feed(b"\n"));
    }

    #[test]
    fn multiple_data_lines_are_one_nonterminal_event() {
        for input in [
            &b"data: [DONE]\ndata: extra\n\n"[..],
            &b"data: extra\ndata: [DONE]\n\n"[..],
            &b"data: [DONE]\ndata\n\n"[..],
            &b": data: [DONE]\n\n"[..],
            &b"data: [DONE] extra\n\n"[..],
        ] {
            let mut parser = DoneEvent::default();
            assert!(!parser.feed(input));
            assert!(parser.feed(b"data:[DONE]\n\n"));
        }
    }

    #[test]
    fn bom_and_all_sse_line_endings() {
        for ending in ["\n", "\r", "\r\n"] {
            let input = format!("\u{feff}data:[DONE]{ending}{ending}");
            let mut parser = DoneEvent::default();
            for byte in input.as_bytes() {
                parser.feed(&[*byte]);
            }
            assert!(parser.completed);
        }
    }

    #[test]
    fn large_payload_stays_bounded_and_done_can_follow_in_same_chunk() {
        let mut input = b"data: {\"text\":\"".to_vec();
        input.extend(vec![b'x'; 1_000_000]);
        input.extend_from_slice(b"data: [DONE]\"}\n\ndata: [DONE]\n\n");
        let mut parser = DoneEvent::default();
        assert!(parser.feed(&input));
        assert!(parser.len <= 17);
    }
}
