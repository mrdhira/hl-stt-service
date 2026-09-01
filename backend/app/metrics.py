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


#: Orthographic spelling -> spoken form, applied to BOTH sides before any
#: comparison so a name written one way and said another is not scored as an
#: error.
#:
#: Why this exists: the speaker's name is written "Dhira" but *pronounced*
#: "Dira" — the h is silent. The models transcribe what they hear and emit
#: "Dira", so ground truth spelled "Dhira" charged one substitution on every
#: utterance containing the name (run #17: 12.5% WER from that single word, on
#: an otherwise perfect transcript). That measures spelling, not recognition.
#:
#: This is a deliberately *explicit* lookup, not fuzzy matching. There is no
#: edit-distance threshold and no phonetic algorithm — those would silently
#: hide real recognition errors. Only the exact whole words listed here are
#: equated, so adding one is a conscious decision.
#:
#: Keys must be lowercase. The frontend keeps an identical map in
#: `frontend/src/lib/wer.ts` — change both together.
PRONUNCIATION_EQUIV: dict[str, str] = {
    "dhira": "dira",
}

# A "word" for equivalence purposes: a run of letters, no digits or
# punctuation, so only whole tokens are rewritten. Substring matching would
# turn an unrelated word like "dhirama" into "dirama".
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def apply_pronunciation_equiv(text: str) -> str:
    """Rewrite orthographic spellings to their spoken form.

    Whole-word only and case-insensitive on the way in. The replacement is
    emitted in the map's own lowercase form, which is fine because every caller
    is counting or comparing rather than displaying.
    """
    if not text or not PRONUNCIATION_EQUIV:
        return text

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        return PRONUNCIATION_EQUIV.get(word.lower(), word)

    return _WORD.sub(replace, text)


def count_words(text: str) -> int:
    """Language-aware-ish token count for English + Japanese.

    Pronunciation equivalences are applied first so the count matches the text
    the WER comparison actually sees.
    """
    if not text:
        return 0
    text = apply_pronunciation_equiv(text)
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
