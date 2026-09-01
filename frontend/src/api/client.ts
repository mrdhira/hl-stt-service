/** Thin fetch wrappers over the backend. No client-side caching — this is a
 *  benchmark tool, stale numbers would be worse than a round trip. */

import { apiUrl } from './config'
import type {
  HealthResponse,
  ModelsResponse,
  RunRow,
  RunsResponse,
  TranscribeResult,
} from './types'

/** An API error carrying the backend's `detail` payload, whatever its shape. */
export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown

  constructor(status: number, detail: unknown) {
    super(formatDetail(status, detail))
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

function formatDetail(status: number, detail: unknown): string {
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object') {
    const record = detail as Record<string, unknown>
    // POST /transcribe returns a structured 503 for an unavailable model.
    if (typeof record.message === 'string') {
      const missing = Array.isArray(record.missing_files) ? record.missing_files : []
      return missing.length > 0
        ? `${record.message} (missing: ${missing.join(', ')})`
        : record.message
    }
    try {
      return JSON.stringify(detail)
    } catch {
      /* fall through */
    }
  }
  return `request failed with status ${status}`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(apiUrl(path), init)
  } catch (cause) {
    throw new ApiError(0, `cannot reach the backend at ${apiUrl(path)}`)
  }

  if (!response.ok) {
    let detail: unknown = null
    try {
      detail = (await response.json()).detail
    } catch {
      detail = await response.text().catch(() => null)
    }
    throw new ApiError(response.status, detail)
  }
  return (await response.json()) as T
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>('/health', { signal })
}

export function getModels(rescan = false, signal?: AbortSignal): Promise<ModelsResponse> {
  return request<ModelsResponse>(`/models${rescan ? '?rescan=true' : ''}`, { signal })
}

export interface RunsQuery {
  limit?: number
  model?: string
  mode?: string
}

export function getRuns(query: RunsQuery = {}, signal?: AbortSignal): Promise<RunsResponse> {
  const params = new URLSearchParams()
  if (query.limit != null) params.set('limit', String(query.limit))
  if (query.model) params.set('model', query.model)
  if (query.mode) params.set('mode', query.mode)
  const suffix = params.toString()
  return request<RunsResponse>(`/runs${suffix ? `?${suffix}` : ''}`, { signal })
}

/**
 * PATCH /runs/{id} — correct a run's ground truth after the fact.
 *
 * Blank clears the label: the backend stores NULL rather than "", so a run is
 * either labelled or it is not, and WER stays undefined for the unlabelled.
 */
export function patchRunExpectedText(
  runId: number,
  expectedText: string,
  signal?: AbortSignal,
): Promise<RunRow> {
  return request<RunRow>(`/runs/${runId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ expected_text: expectedText.trim() }),
    signal,
  })
}

/**
 * POST /transcribe. The backend takes WAV only and reads the file from the
 * multipart field named `audio`; `expected_text` is an optional sibling field.
 */
export function postTranscribe(
  model: string,
  wav: Blob,
  expectedText?: string,
  signal?: AbortSignal,
): Promise<TranscribeResult> {
  const body = new FormData()
  body.append('audio', wav, 'clip.wav')
  const trimmed = expectedText?.trim()
  if (trimmed) body.append('expected_text', trimmed)

  return request<TranscribeResult>(`/transcribe?model=${encodeURIComponent(model)}`, {
    method: 'POST',
    body,
    signal,
  })
}
