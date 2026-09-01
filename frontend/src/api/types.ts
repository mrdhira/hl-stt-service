/**
 * TypeScript mirrors of backend/app/schemas.py.
 *
 * Field names and nullability follow the pydantic models exactly — if the
 * backend schema changes, change it here too.
 */

/** `ModelInfo` — availability of one recognizer. */
export interface ModelInfo {
  key: string
  display_name: string
  available: boolean
  loaded: boolean
  /** False for all three POC models: sherpa-onnx has no OnlineRecognizer for them. */
  supports_streaming: boolean
  factory: string
  directory: string
  missing_files: string[]
  error: string | null
}

/** `ModelsResponse` — GET /models */
export interface ModelsResponse {
  sherpa_onnx_available: boolean
  sherpa_onnx_version: string | null
  models_dir: string
  num_threads: number
  models: ModelInfo[]
}

/** `TranscribeResult` — POST /transcribe */
export interface TranscribeResult {
  model: string
  /** "batch" | "stream" */
  mode: string
  text: string
  audio_ms: number
  processing_ms: number
  rtf: number
  words: number
  chars: number
  text_hash: string
  sample_rate: number
  latency_partial_ms: number | null
  latency_final_ms: number | null
  expected_text: string | null
  run_id: number | null
}

/** `RunRow` — one row of the `runs` table. */
export interface RunRow {
  id: number
  ts: number
  model: string
  mode: string
  audio_ms: number | null
  processing_ms: number | null
  rtf: number | null
  words: number | null
  chars: number | null
  latency_partial_ms: number | null
  latency_final_ms: number | null
  text: string | null
  text_hash: string | null
  expected_text: string | null
  /** Where the raw upload was archived as training data, if it was. */
  audio_path: string | null
}

/** `RunsResponse` — GET /runs */
export interface RunsResponse {
  count: number
  runs: RunRow[]
}

/** `AsrResult` — POST /asr, the Hermes voice-note endpoint. */
export interface AsrResult {
  text: string
  model: string
  audio_ms: number
  processing_ms: number
  rtf: number
  run_id: number | null
}

/** GET /health */
export interface HealthResponse {
  status: string
  version: string
  sherpa_onnx: string | null
  models_available: string[]
}

// --- WS /stream frames -------------------------------------------------------
// Mirrors the wire protocol documented in backend/app/routes/stream.py.

export interface StreamReady {
  type: 'ready'
  model: string
  mode: string
  supports_true_streaming: boolean
  partial_interval_ms: number
  sample_rate: number
  note: string
}

export interface StreamPartial {
  type: 'partial'
  text: string
  audio_ms: number
  processing_ms: number
  elapsed_ms: number
}

export interface StreamFinal {
  type: 'final'
  model: string
  mode: string
  text: string
  audio_ms: number
  /** Sum of every decode, partials included — not comparable to batch RTF. */
  processing_ms: number
  final_decode_ms: number
  rtf: number
  words: number
  chars: number
  text_hash: string
  sample_rate: number
  partials_emitted: number
  latency_partial_ms: number | null
  latency_final_ms: number | null
  expected_text: string | null
  run_id: number | null
}

export interface StreamError {
  type: 'error'
  detail: string
}

export interface StreamReset {
  type: 'reset'
}

export type StreamMessage =
  | StreamReady
  | StreamPartial
  | StreamFinal
  | StreamError
  | StreamReset
