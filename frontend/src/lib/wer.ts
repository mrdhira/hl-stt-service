/**
 * Word error rate.
 *
 * Tokenisation follows `count_words` in backend/app/metrics.py: Japanese has no
 * spaces, so each CJK/kana character is its own token and the rest is split on
 * whitespace. Both sides apply the same NFKC + lowercase + PRONUNCIATION_EQUIV
 * normalisation first, so the same input yields the same tokens.
 *
 * They are close but NOT identical, and the difference is punctuation: this
 * tokenizer strips it before splitting, `count_words` does not. So
 * `count_words("dhira-san")` is 1 while `tokenize("dhira-san")` is 2. Treat the
 * `words` column as an approximate size and the WER denominator
 * (`referenceLength`) as the authoritative token count for scoring.
 */

// CJK ideographs, hiragana, katakana, halfwidth katakana — same ranges as the
// backend regex.
const CJK = /[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]/gu

// Punctuation is not a transcription error; strip it before comparing.
const PUNCTUATION = /[.,!?;:"'“”‘’()[\]{}<>«»…—–\-。、！？；：「」『』（）]/gu

/**
 * Orthographic spelling -> spoken form, applied to BOTH sides before any
 * comparison so a name written one way and said another is not scored as an
 * error.
 *
 * Why this exists: the speaker's name is written "Dhira" but *pronounced*
 * "Dira" — the h is silent. The models transcribe what they hear and emit
 * "Dira", so ground truth spelled "Dhira" charged one substitution on every
 * utterance containing the name (run #17: 12.5% WER from that single word, on
 * an otherwise perfect transcript). That measures spelling, not recognition.
 *
 * This is a deliberately *explicit* lookup, not fuzzy matching. There is no
 * edit-distance threshold and no phonetic algorithm — those would silently
 * hide real recognition errors. Only the exact whole words listed here are
 * equated, so adding one is a conscious decision.
 *
 * Keys must be lowercase. The backend keeps an identical map in
 * `backend/app/metrics.py` — change both together.
 */
export const PRONUNCIATION_EQUIV: Record<string, string> = {
  dhira: 'dira',
}

// A "word" for equivalence purposes: a run of letters, no digits or
// punctuation, so only whole tokens are rewritten. Substring matching would
// turn an unrelated word like "dhirama" into "dirama".
//
// Reconciled with the backend's `[^\W\d_]+`: Python excludes digits and
// underscore explicitly and matches letters only — combining marks are not
// word characters there either — so the two classes agree. Verified on
// fullwidth, decomposed-accent, digit-suffixed and underscore-joined inputs.
const WORD = /\p{L}+/gu

/** Rewrite orthographic spellings to their spoken form. Whole words only. */
export function applyPronunciationEquiv(text: string): string {
  if (!text) return text
  return text.replace(WORD, (word) => PRONUNCIATION_EQUIV[word.toLowerCase()] ?? word)
}

export function tokenize(text: string): string[] {
  if (!text) return []
  // Equivalences run after NFKC + lowercase and before the token split, so the
  // map only ever needs lowercase keys.
  const normalised = applyPronunciationEquiv(
    text.normalize('NFKC').toLowerCase(),
  ).replace(PUNCTUATION, ' ')

  const tokens: string[] = []
  for (const chunk of normalised.split(/\s+/)) {
    if (!chunk) continue
    // Split a mixed chunk into CJK singles plus latin runs.
    let latin = ''
    for (const char of chunk) {
      CJK.lastIndex = 0
      if (CJK.test(char)) {
        if (latin) {
          tokens.push(latin)
          latin = ''
        }
        tokens.push(char)
      } else {
        latin += char
      }
    }
    if (latin) tokens.push(latin)
  }
  return tokens
}

export interface WerResult {
  /** (substitutions + deletions + insertions) / reference length. */
  wer: number
  substitutions: number
  deletions: number
  insertions: number
  hits: number
  /** Reference token count — the denominator. */
  referenceLength: number
}

/**
 * Levenshtein alignment over tokens, with backtracking for the S/D/I split.
 *
 * O(n·m) time, O(n·m) memory. Benchmark clips are seconds long, so the matrix
 * stays tiny; if that ever changes, switch to a banded implementation.
 */
export function wordErrorRate(reference: string, hypothesis: string): WerResult | null {
  const ref = tokenize(reference)
  const hyp = tokenize(hypothesis)

  // No ground truth means no WER to report — not "0% error".
  if (ref.length === 0) return null

  const rows = ref.length + 1
  const cols = hyp.length + 1
  const cost = new Uint32Array(rows * cols)

  for (let i = 0; i < rows; i += 1) cost[i * cols] = i
  for (let j = 0; j < cols; j += 1) cost[j] = j

  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      const substitute = cost[(i - 1) * cols + (j - 1)] + (ref[i - 1] === hyp[j - 1] ? 0 : 1)
      const deleted = cost[(i - 1) * cols + j] + 1
      const inserted = cost[i * cols + (j - 1)] + 1
      cost[i * cols + j] = Math.min(substitute, deleted, inserted)
    }
  }

  let substitutions = 0
  let deletions = 0
  let insertions = 0
  let hits = 0
  let i = ref.length
  let j = hyp.length

  while (i > 0 || j > 0) {
    const current = cost[i * cols + j]
    if (i > 0 && j > 0) {
      const same = ref[i - 1] === hyp[j - 1]
      if (current === cost[(i - 1) * cols + (j - 1)] + (same ? 0 : 1)) {
        if (same) hits += 1
        else substitutions += 1
        i -= 1
        j -= 1
        continue
      }
    }
    if (i > 0 && current === cost[(i - 1) * cols + j] + 1) {
      deletions += 1
      i -= 1
      continue
    }
    insertions += 1
    j -= 1
  }

  return {
    // Can exceed 1.0 when the hypothesis is longer than the reference — that is
    // correct WER behaviour, not a bug to clamp away.
    wer: (substitutions + deletions + insertions) / ref.length,
    substitutions,
    deletions,
    insertions,
    hits,
    referenceLength: ref.length,
  }
}
