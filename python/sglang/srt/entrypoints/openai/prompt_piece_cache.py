"""Tokenize rendered chat prompts piece by piece, reusing the pieces earlier turns already encoded.

A rendered multi-turn prompt is the previous turn's prompt plus a few new messages, so encoding the
whole string every turn is O(conversation) per request (about 0.5 s at 186K tokens). The tokenizer
splits its input at added tokens before normalization and pre-tokenization, so encoding the text
between special added tokens separately gives the same ids; each such piece is cached by content.
"""

import re
from array import array
from collections import OrderedDict
from typing import List, Optional


def _split_pattern(tokenizer) -> Optional[re.Pattern]:
    """Special added tokens that are safe split points, or None when the tokenizer has none."""
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if not decoder or tokenizer.encode("") != []:
        # A post-processor that adds ids around the whole text cannot be applied per piece.
        return None
    added = [tok.content for tok in decoder.values()]
    safe = [
        tok.content
        for tok in decoder.values()
        if tok.special
        and not (tok.lstrip or tok.rstrip or tok.single_word or tok.normalized)
        # A split token inside a longer added token would split where the tokenizer does not.
        and not any(tok.content in other and tok.content != other for other in added)
    ]
    if not safe:
        return None
    safe.sort(key=len, reverse=True)
    return re.compile("(" + "|".join(re.escape(s) for s in safe) + ")")


class PromptPieceEncoder:
    def __init__(self, tokenizer, max_cached_tokens: int):
        self._tokenizer = tokenizer
        self._max_cached_tokens = max_cached_tokens
        self._pattern = _split_pattern(tokenizer) if max_cached_tokens > 0 else None
        self._cache: "OrderedDict[str, array]" = OrderedDict()
        self._cached_tokens = 0

    @property
    def enabled(self) -> bool:
        return self._pattern is not None

    def encode(self, text: str) -> List[int]:
        if self._pattern is None:
            return self._tokenizer.encode(text)
        ids: List[int] = []
        for piece in self._pattern.split(text):
            if not piece:
                continue
            piece_ids = self._cache.get(piece)
            if piece_ids is None:
                piece_ids = array("i", self._tokenizer.encode(piece, add_special_tokens=False))
                self._insert(piece, piece_ids)
            else:
                self._cache.move_to_end(piece)
            ids.extend(piece_ids)
        return ids

    def _insert(self, piece: str, piece_ids: array) -> None:
        if len(piece_ids) > self._max_cached_tokens:
            return
        self._cache[piece] = piece_ids
        self._cached_tokens += len(piece_ids)
        while self._cached_tokens > self._max_cached_tokens:
            _, evicted = self._cache.popitem(last=False)
            self._cached_tokens -= len(evicted)
