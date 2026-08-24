import type { ModelInfo } from '../api/types'

interface Props {
  models: ModelInfo[]
  value: string
  onChange: (key: string) => void
  disabled?: boolean
  /** Show only models the backend reports as streaming-capable. */
  requireStreaming?: boolean
}

/**
 * Model dropdown. Unavailable models stay visible but disabled, with the reason
 * spelled out — a silently missing option looks like a bug, a disabled one with
 * "no model files" tells you to go download it.
 */
export function ModelPicker({ models, value, onChange, disabled, requireStreaming }: Props) {
  const selected = models.find((m) => m.key === value)

  return (
    <div className="stack" style={{ gap: 8, minWidth: 260 }}>
      <label className="field">
        Model
        <select
          value={value}
          disabled={disabled || models.length === 0}
          onChange={(event) => onChange(event.target.value)}
        >
          {models.length === 0 && <option value="">no models reported</option>}
          {models.map((model) => (
            <option key={model.key} value={model.key} disabled={!model.available}>
              {model.display_name}
              {model.available ? (model.loaded ? ' · loaded' : ' · ready') : ' · unavailable'}
            </option>
          ))}
        </select>
      </label>

      {selected && !selected.available && (
        <div className="notice warn">
          {selected.error ?? 'Model files not found.'}
          {selected.missing_files.length > 0 && (
            <>
              {' '}Missing <span className="mono">{selected.missing_files.join(', ')}</span> in{' '}
              <span className="mono">{selected.directory}</span>.
            </>
          )}
        </div>
      )}

      {selected?.available && requireStreaming && !selected.supports_streaming && (
        <div className="notice">
          <strong>Pseudo-streaming.</strong> sherpa-onnx has no streaming recognizer for{' '}
          {selected.display_name}, so the backend re-decodes the whole buffer on an
          interval. Partials may change between emissions, and stream RTF is not
          comparable to batch RTF.
        </div>
      )}
    </div>
  )
}
