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

# A "word" for equivalence purposes: a run of letters, no digits, underscores
# or punctuation, so only whole tokens are rewritten. Substring matching would
# turn an unrelated word like "dhirama" into "dirama".
#
# The frontend uses /\p{L}+/u for the same job. The two agree: Python's
# ``[^\W\d_]`` is letters only — combining marks and digits are both excluded,
# matching \p{L}. `scripts/migrate_pronunciation.py` reuses this constant so
# the tokenizer cannot drift from the scorer either.
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def normalise_for_compare(text: str) -> str:
    """NFKC + casefold, the shared pre-step before any equivalence matching.

    The frontend does ``normalize('NFKC').toLowerCase()`` before applying its
    map; without the same step here, fullwidth input like ``Ｄｈｉｒａ`` would be
    equated in the UI's WER but not in ``count_words``. Keeping the order
    identical on both sides is what makes the two maps interchangeable.
    """
    return unicodedata.normalize("NFKC", text).lower()


def apply_pronunciation_equiv(text: str) -> str:
    """Rewrite orthographic spellings to their spoken form.

    Whole-word only. Input is NFKC-normalised and lowercased first — exactly
    what the frontend tokenizer does — so the returned text is lowercase. Every
    caller is counting or comparing rather than displaying, so that is fine.
    """
    if not text:
        return text
    text = normalise_for_compare(text)
    if not PRONUNCIATION_EQUIV:
        return text

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        return PRONUNCIATION_EQUIV.get(word, word)

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
    """sha256 of normalised, whitespace-collapsed text.

    Stable across runs so identical transcripts of the same audio collapse to
    one hash when comparing models. Pronunciation equivalences are applied for
    the same reason they are applied to `count_words`: a model that writes
    "Dira" and one that writes "Dhira" said the same thing, and hashing them
    apart while `words` treats them as equal would be inconsistent.

    Note this changes the hash of any text containing a mapped word, so hashes
    stored before an entry was added will not match freshly computed ones. The
    column is a comparison aid, not a key, so that is acceptable.
    """
    normalised = apply_pronunciation_equiv(text or "").strip()
    normalised = " ".join(normalised.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def rtf(processing_ms: float, audio_ms: float) -> float:
    """Real-time factor. 0.0 when the audio has no duration."""
    if audio_ms <= 0:
        return 0.0
    return processing_ms / audio_ms
