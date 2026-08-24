/**
 * Word error rate.
 *
 * Tokenisation mirrors `count_words` in backend/app/metrics.py: Japanese has no
 * spaces, so each CJK/kana character is its own token and the rest is split on
 * whitespace. Keeping the two in sync means the `words` column and the WER
 * denominator count the same things.
 */

// CJK ideographs, hiragana, katakana, halfwidth katakana — same ranges as the
// backend regex.
const CJK = /[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]/gu

// Punctuation is not a transcription error; strip it before comparing.
const PUNCTUATION = /[.,!?;:"'“”‘’()[\]{}<>«»…—–\-。、！？；：「」『』（）]/gu

export function tokenize(text: string): string[] {
  if (!text) return []
  const normalised = text.normalize('NFKC').toLowerCase().replace(PUNCTUATION, ' ')

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
