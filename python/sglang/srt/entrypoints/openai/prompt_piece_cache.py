"""Tokenize rendered chat prompts piece by piece, reusing the pieces earlier turns already encoded.

A rendered multi-turn prompt is the previous turn's prompt plus a few new messages, so encoding the
whole string every turn is O(conversation) per request (about 0.2 s at 186K tokens on GB300). The
tokenizer splits its input at added tokens before normalization and pre-tokenization, so encoding the
text between special added tokens separately gives the same ids; each such piece is cached by content.
Every seam next to a newly encoded piece is re-encoded as one window and checked; a mismatch falls
back to a full encode of that prompt and stops splitting on that token.
"""

import logging
import re
import threading
from array import array
from collections import OrderedDict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Characters of each neighbour re-encoded with the split token when checking a seam.
_SEAM_CONTEXT_CHARS = 32


def _safe_split_tokens(tokenizer) -> Dict[str, int]:
    """Special added tokens that are safe split points (content -> id); empty when there are none."""
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if not decoder or tokenizer.encode("") != []:
        # A post-processor that adds ids around the whole text cannot be applied per piece.
        return {}
    added = [tok.content for tok in decoder.values()]
    return {
        tok.content: token_id
        for token_id, tok in decoder.items()
        if tok.special
        and not (tok.lstrip or tok.rstrip or tok.single_word or tok.normalized)
        # A split token inside a longer added token would split where the tokenizer does not.
        and not any(tok.content in other and tok.content != other for other in added)
    }


def _compile(split_tokens: Dict[str, int]) -> Optional[re.Pattern]:
    if not split_tokens:
        return None
    ordered = sorted(split_tokens, key=len, reverse=True)
    return re.compile("(" + "|".join(re.escape(s) for s in ordered) + ")")


class PromptPieceEncoder:
    def __init__(self, tokenizer, max_cached_tokens: int):
        self._tokenizer = tokenizer
        self._max_cached_tokens = max_cached_tokens
        self._split_tokens = _safe_split_tokens(tokenizer) if max_cached_tokens > 0 else {}
        self._pattern = _compile(self._split_tokens)
        self._cache: "OrderedDict[str, array]" = OrderedDict()
        self._cached_tokens = 0
        # Chat conversion may run on worker threads; the tokenizer itself is called outside the lock.
        self._lock = threading.Lock()
        self.num_fallbacks = 0

    @property
    def enabled(self) -> bool:
        return self._pattern is not None

    def encode(self, text: str) -> List[int]:
        pattern = self._pattern
        if pattern is None:
            return self._tokenizer.encode(text)
        pieces = [p for p in pattern.split(text) if p]
        piece_ids, fresh = self._lookup(pieces)
        if fresh and not self._seams_hold(pieces, fresh):
            return self._tokenizer.encode(text)
        ids: List[int] = []
        for chunk in piece_ids:
            ids.extend(chunk)
        return ids

    def _lookup(self, pieces: List[str]):
        piece_ids, fresh = [], set()
        for i, piece in enumerate(pieces):
            with self._lock:
                hit = self._cache.get(piece)
                if hit is not None:
                    self._cache.move_to_end(piece)
            if hit is None:
                hit = array("i", self._encode_piece(piece))
                fresh.add(i)
            piece_ids.append(hit)
        for i in fresh:
            self._insert(pieces[i], piece_ids[i])
        return piece_ids, fresh

    def _encode_piece(self, piece: str) -> List[int]:
        token_id = self._split_tokens.get(piece)
        if token_id is not None:
            return [token_id]
        return self._tokenizer.encode(piece, add_special_tokens=False)

    def _seams_hold(self, pieces: List[str], fresh: set) -> bool:
        """Re-encode each split token with its neighbours' edge text where a neighbour is new."""
        for i, piece in enumerate(pieces):
            if piece not in self._split_tokens or not ({i - 1, i + 1} & fresh):
                continue
            left = pieces[i - 1][-_SEAM_CONTEXT_CHARS:] if i > 0 and pieces[i - 1] not in self._split_tokens else ""
            right = pieces[i + 1][:_SEAM_CONTEXT_CHARS] if i + 1 < len(pieces) and pieces[i + 1] not in self._split_tokens else ""
            joined = self._tokenizer.encode(left + piece + right, add_special_tokens=False)
            parts = (
                self._tokenizer.encode(left, add_special_tokens=False)
                + [self._split_tokens[piece]]
                + self._tokenizer.encode(right, add_special_tokens=False)
            )
            if joined != parts:
                self._disable_split_token(piece)
                return False
        return True

    def _disable_split_token(self, token: str) -> None:
        with self._lock:
            self.num_fallbacks += 1
            self._split_tokens = {k: v for k, v in self._split_tokens.items() if k != token}
            self._pattern = _compile(self._split_tokens)
            self._cache.clear()
            self._cached_tokens = 0
        logger.warning("Prompt piece cache: seam mismatch at %r; full encode, token no longer split.", token)

    def _insert(self, piece: str, piece_ids: array) -> None:
        if len(piece_ids) > self._max_cached_tokens:
            return
        with self._lock:
            if piece in self._cache:
                return
            self._cache[piece] = piece_ids
            self._cached_tokens += len(piece_ids)
            while self._cached_tokens > self._max_cached_tokens:
                _, evicted = self._cache.popitem(last=False)
                self._cached_tokens -= len(evicted)
