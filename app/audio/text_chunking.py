"""Text chunking helpers for TTS backends."""

from __future__ import annotations

import re


SENTENCE_END_RE = re.compile(r"(.+?[.!?。！？])(?:\s+|$)", re.DOTALL)
CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")
WORD_RE = re.compile(r"\S+")


class SentenceChunker:
    def __init__(self) -> None:
        self._buffer = ""

    def add_chunk(self, text: str) -> list[str]:
        self._buffer += text
        sentences: list[str] = []
        while True:
            match = SENTENCE_END_RE.match(self._buffer)
            if match is None:
                break
            sentence = match.group(1).strip()
            if sentence:
                sentences.append(sentence)
            self._buffer = self._buffer[match.end():]
        return sentences

    def finish(self) -> str:
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder


def split_text_for_tts(text: str, *, max_chars: int = 120) -> list[str]:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return []
    if max_chars <= 0 or len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    chunker = SentenceChunker()

    def append_piece(piece: str) -> None:
        stripped = piece.strip()
        if not stripped:
            return
        if len(stripped) <= max_chars:
            chunks.append(stripped)
            return

        clauses = [part.strip() for part in CLAUSE_SPLIT_RE.split(stripped) if part.strip()]
        if len(clauses) > 1:
            current = ""
            for clause in clauses:
                if len(clause) > max_chars:
                    if current:
                        chunks.append(current)
                        current = ""
                    append_piece(clause)
                    continue
                candidate = clause if not current else f"{current} {clause}"
                if current and len(candidate) > max_chars:
                    chunks.append(current)
                    current = clause
                else:
                    current = candidate
            if current:
                chunks.append(current)
            return

        current = ""
        for word in stripped.split():
            candidate = word if not current else f"{current} {word}"
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current:
            chunks.append(current)

    for sentence in chunker.add_chunk(normalized):
        append_piece(sentence)

    remainder = chunker.finish()
    if remainder:
        append_piece(remainder)

    return chunks


def split_text_for_realtime_tts(
    text: str,
    *,
    max_chars: int = 120,
    first_fragment_min_words: int = 6,
    first_fragment_max_words: int = 12,
    first_fragment_delimiters: str = ".?!;:,",
) -> list[str]:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return []

    first_fragment, remainder = _split_first_fragment(
        normalized,
        min_words=max(1, first_fragment_min_words),
        max_words=max(0, first_fragment_max_words),
        delimiters=first_fragment_delimiters,
    )
    if first_fragment is None:
        return split_text_for_tts(normalized, max_chars=max_chars)

    chunks = split_text_for_tts(first_fragment, max_chars=max_chars)
    if remainder:
        chunks.extend(split_text_for_tts(remainder, max_chars=max_chars))
    return chunks


def _split_first_fragment(
    text: str,
    *,
    min_words: int,
    max_words: int,
    delimiters: str,
) -> tuple[str | None, str]:
    if max_words <= 0:
        return None, text

    words = list(WORD_RE.finditer(text))
    if len(words) <= max_words:
        return None, text

    min_index = words[min(min_words, len(words)) - 1].end()
    max_index = words[max_words - 1].end()

    boundary = None
    for index in range(min_index, min(max_index + 1, len(text))):
        if text[index] in delimiters:
            boundary = index + 1
            break

    if boundary is None:
        boundary = max_index

    while boundary < len(text) and text[boundary].isspace():
        boundary += 1

    first = text[:boundary].strip()
    remainder = text[boundary:].strip()
    if not first or not remainder:
        return None, text
    return first, remainder
