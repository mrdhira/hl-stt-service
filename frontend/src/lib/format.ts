/** Display formatting. Kept in one place so the tabs read consistently. */

export function ms(value: number | null | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return '—'
  if (value >= 10000) return `${(value / 1000).toFixed(2)} s`
  return `${value.toFixed(digits)} ms`
}

export function rtf(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—'
  return value.toFixed(3)
}

export function count(value: number | null | undefined): string {
  return value == null ? '—' : String(value)
}

export function percent(value: number | null | undefined, digits = 1): string {
  if (value == null || !Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

/** Unix seconds (the `ts` column) as a local timestamp. */
export function timestamp(seconds: number): string {
  const date = new Date(seconds * 1000)
  return date.toLocaleString(undefined, {
    year: '2-digit',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

/** Colour band for an RTF value: under 1.0 is faster than realtime. */
export function rtfTone(value: number | null | undefined): 'good' | 'warn' | 'bad' | 'none' {
  if (value == null || !Number.isFinite(value)) return 'none'
  if (value < 0.5) return 'good'
  if (value < 1) return 'warn'
  return 'bad'
}

/** Colour band for WER. */
export function werTone(value: number | null | undefined): 'good' | 'warn' | 'bad' | 'none' {
  if (value == null || !Number.isFinite(value)) return 'none'
  if (value <= 0.1) return 'good'
  if (value <= 0.3) return 'warn'
  return 'bad'
}
