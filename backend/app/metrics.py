"""Benchmark metrics derived from a decode."""

from __future__ import annotations

import hashlib
import re
import unicodedata

# CJK ideographs, hiragana, katakana, and halfwidth katakana. Japanese has no
# spaces, so whitespace splitting would report ~1 word per utterance; we count
# each CJK/kana character as one token instead and add the whitespace-separated
# tokens from the rest.
_CJK = re.compile(
    r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]"
)


def count_words(text: str) -> int:
    """Language-aware-ish token count for English + Japanese."""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    latin = len(_CJK.sub(" ", text).split())
    return cjk + latin


def count_chars(text: str) -> int:
    return len(text or "")


def text_hash(text: str) -> str:
    """sha256 of NFKC-normalised, whitespace-collapsed text.

    Stable across runs so identical transcripts of the same audio collapse to
    one hash when comparing models.
    """
    normalised = unicodedata.normalize("NFKC", text or "").strip()
    normalised = " ".join(normalised.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def rtf(processing_ms: float, audio_ms: float) -> float:
    """Real-time factor. 0.0 when the audio has no duration."""
    if audio_ms <= 0:
        return 0.0
    return processing_ms / audio_ms
