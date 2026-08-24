/** Runtime config, all from import.meta.env. No secrets live here. */

function str(value: unknown, fallback: string): string {
  return typeof value === 'string' && value.length > 0 ? value : fallback
}

function num(value: unknown, fallback: number): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

/** Empty string = same origin (dev proxy, or Caddy in production). */
export const API_BASE = str(import.meta.env.VITE_API_BASE, '').replace(/\/$/, '')

export const STREAM_SAMPLE_RATE = num(import.meta.env.VITE_STREAM_SAMPLE_RATE, 16000)

export const RUNS_LIMIT = num(import.meta.env.VITE_RUNS_LIMIT, 100)

/** Absolute URL for an API path. */
export function apiUrl(path: string): string {
  return `${API_BASE}${path}`
}

/** ws:// or wss:// URL for the websocket endpoint. */
export function wsUrl(path: string): string {
  const base = API_BASE || window.location.origin
  const url = new URL(API_BASE ? `${base}${path}` : path, window.location.origin)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}
