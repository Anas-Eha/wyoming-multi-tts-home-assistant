from __future__ import annotations

from app.audio.text_chunking import split_text_for_realtime_tts, split_text_for_tts


def test_split_text_for_tts_breaks_long_text() -> None:
    chunks = split_text_for_tts(
        "To jest pierwsze zdanie. To jest drugie zdanie, ktore jest troche dluzsze i nadal powinno byc poprawnie dzielone.",
        max_chars=50,
    )
    assert len(chunks) >= 2
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_split_text_for_realtime_tts_prefers_short_first_fragment() -> None:
    chunks = split_text_for_realtime_tts(
        "To jest pierwsza czesc testu, ktora ma byc krotka. A to jest druga czesc zdania do dalszej syntezy.",
        max_chars=120,
        first_fragment_min_words=3,
        first_fragment_max_words=8,
    )
    assert len(chunks) >= 2
    assert chunks[0].startswith("To jest")
