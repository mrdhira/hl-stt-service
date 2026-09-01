import { useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, patchRunExpectedText } from '../api/client'
import { RUNS_LIMIT } from '../api/config'
import type { ModelInfo, RunRow } from '../api/types'
import { count, ms, percent, rtf, rtfTone, timestamp, werTone } from '../lib/format'
import { wordErrorRate } from '../lib/wer'
import type { WerResult } from '../lib/wer'

interface Props {
  models: ModelInfo[]
  runs: RunRow[]
  loading: boolean
  error: string | null
  limit: number
  onLimitChange: (limit: number) => void
  onRefresh: () => void
}

interface ScoredRun extends RunRow {
  wer: WerResult | null
}

/** Per-model roll-up shown above the table. */
interface Summary {
  model: string
  runs: number
  meanRtf: number | null
  meanWer: number | null
  scored: number
}

export function Reports({
  models,
  runs,
  loading,
  error,
  limit,
  onLimitChange,
  onRefresh,
}: Props) {
  const [modelFilter, setModelFilter] = useState('')
  const [modeFilter, setModeFilter] = useState('')
  const [taggedOnly, setTaggedOnly] = useState(false)

  // Inline ground-truth editing. One row at a time: opening another editor
  // while one is dirty would need a confirm dialog for no real benefit.
  const [editingId, setEditingId] = useState<number | null>(null)
  const [savingId, setSavingId] = useState<number | null>(null)
  const [editError, setEditError] = useState<string | null>(null)

  const beginEdit = (run: RunRow) => {
    setEditingId(run.id)
    setEditError(null)
  }

  const cancelEdit = () => {
    setEditingId(null)
    setEditError(null)
  }

  const saveEdit = async (runId: number, value: string) => {
    setSavingId(runId)
    setEditError(null)
    try {
      await patchRunExpectedText(runId, value)
      setEditingId(null)
      // Refetch rather than patching local state: WER and the per-model
      // roll-up both derive from the row, and the server is the source of
      // truth for what was actually stored (blank becomes null).
      onRefresh()
    } catch (cause) {
      setEditError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setSavingId(null)
    }
  }

  const scored: ScoredRun[] = useMemo(
    () =>
      runs.map((run) => ({
        ...run,
        // WER is computed here, not stored: the tokenizer can change without a
        // migration, and the ground truth is on the row.
        wer: run.expected_text ? wordErrorRate(run.expected_text, run.text ?? '') : null,
      })),
    [runs],
  )

  const visible = useMemo(
    () =>
      scored.filter(
        (run) =>
          (!modelFilter || run.model === modelFilter) &&
          (!modeFilter || run.mode === modeFilter) &&
          (!taggedOnly || run.expected_text != null),
      ),
    [scored, modelFilter, modeFilter, taggedOnly],
  )

  const summaries = useMemo(() => summarize(visible), [visible])

  const knownModels = useMemo(() => {
    const keys = new Set(models.map((m) => m.key))
    runs.forEach((run) => keys.add(run.model))
    return [...keys]
  }, [models, runs])

  return (
    <div className="stack">
      <section className="panel">
        <h2>Filters</h2>
        <div className="row">
          <label className="field" style={{ minWidth: 180 }}>
            Model
            <select value={modelFilter} onChange={(e) => setModelFilter(e.target.value)}>
              <option value="">All models</option>
              {knownModels.map((key) => (
                <option key={key} value={key}>
                  {models.find((m) => m.key === key)?.display_name ?? key}
                </option>
              ))}
            </select>
          </label>

          <label className="field" style={{ minWidth: 150 }}>
            Mode
            <select value={modeFilter} onChange={(e) => setModeFilter(e.target.value)}>
              <option value="">All modes</option>
              <option value="batch">batch</option>
              <option value="stream">stream</option>
              <option value="asr">asr</option>
            </select>
          </label>

          <label className="field" style={{ minWidth: 110 }}>
            Limit
            <select value={limit} onChange={(e) => onLimitChange(Number(e.target.value))}>
              {[25, 50, 100, 250, 500, 1000].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>

          <label className="row" style={{ gap: 6, alignSelf: 'flex-end', paddingBottom: 8 }}>
            <input
              type="checkbox"
              checked={taggedOnly}
              onChange={(e) => setTaggedOnly(e.target.checked)}
            />
            <span className="dim" style={{ fontSize: 13 }}>
              Test cases only
            </span>
          </label>

          <button
            className="btn"
            onClick={onRefresh}
            disabled={loading}
            style={{ alignSelf: 'flex-end', marginBottom: 2 }}
          >
            {loading ? 'Loading…' : 'Refresh'}
          </button>
        </div>
      </section>

      {error && <div className="notice error">{error}</div>}

      {summaries.length > 0 && (
        <section className="panel">
          <h2>Per model · {visible.length} run{visible.length === 1 ? '' : 's'}</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th style={{ textAlign: 'right' }}>Runs</th>
                  <th style={{ textAlign: 'right' }}>Mean RTF</th>
                  <th style={{ textAlign: 'right' }}>Mean WER</th>
                  <th style={{ textAlign: 'right' }}>Scored</th>
                </tr>
              </thead>
              <tbody>
                {summaries.map((row) => (
                  <tr key={row.model}>
                    <td>{models.find((m) => m.key === row.model)?.display_name ?? row.model}</td>
                    <td className="num">{row.runs}</td>
                    <td className={`num tone-${rtfTone(row.meanRtf)}`}>{rtf(row.meanRtf)}</td>
                    <td className={`num tone-${werTone(row.meanWer)}`}>{percent(row.meanWer)}</td>
                    <td className="num dim">{row.scored}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="faint" style={{ marginTop: 10 }}>
            Mean WER covers only runs tagged with expected text. Mixing modes skews
            mean RTF — a <span className="mono">stream</span> row sums every partial
            decode, so filter to one mode before comparing.
          </div>
        </section>
      )}

      <section className="panel">
        <h2>Runs</h2>
        {visible.length === 0 ? (
          <div className="empty">
            {loading
              ? 'Loading runs…'
              : runs.length === 0
                ? 'No runs yet. Record a clip on the Record & Send tab and transcribe it.'
                : 'No runs match these filters.'}
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Model</th>
                  <th>Mode</th>
                  <th style={{ textAlign: 'right' }}>RTF</th>
                  <th style={{ textAlign: 'right' }}>Audio</th>
                  <th style={{ textAlign: 'right' }}>Proc</th>
                  <th style={{ textAlign: 'right' }}>Words</th>
                  <th style={{ textAlign: 'right' }}>Chars</th>
                  <th style={{ textAlign: 'right' }}>1st part.</th>
                  <th style={{ textAlign: 'right' }}>Final lat.</th>
                  <th style={{ textAlign: 'right' }}>WER</th>
                  <th>Model said</th>
                  <th>Expected (ground truth)</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((run) => (
                  <tr key={run.id}>
                    <td className="dim mono" style={{ whiteSpace: 'nowrap' }}>
                      {timestamp(run.ts)}
                    </td>
                    <td>{run.model}</td>
                    <td className="dim">{run.mode}</td>
                    <td className={`num tone-${rtfTone(run.rtf)}`}>{rtf(run.rtf)}</td>
                    <td className="num dim">{ms(run.audio_ms)}</td>
                    <td className="num dim">{ms(run.processing_ms)}</td>
                    <td className="num">{count(run.words)}</td>
                    <td className="num">{count(run.chars)}</td>
                    <td className="num dim">{ms(run.latency_partial_ms)}</td>
                    <td className="num dim">{ms(run.latency_final_ms)}</td>
                    <td
                      className={`num tone-${werTone(run.wer?.wer)}`}
                      title={
                        run.wer
                          ? `${run.wer.substitutions}S ${run.wer.deletions}D ${run.wer.insertions}I of ${run.wer.referenceLength} tokens`
                          : 'no expected text on this run'
                      }
                    >
                      {run.wer ? percent(run.wer.wer) : '—'}
                    </td>
                    <td className="text">
                      {run.text || <span className="faint">(empty)</span>}
                    </td>
                    <td className="text">
                      <ExpectedCell
                        run={run}
                        editing={editingId === run.id}
                        onEdit={() => beginEdit(run)}
                        onCancel={cancelEdit}
                        onSave={saveEdit}
                        saving={savingId === run.id}
                        error={editingId === run.id ? editError : null}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="faint" style={{ marginTop: 10 }}>
          Showing up to {limit} rows (default {RUNS_LIMIT}), newest first.
        </div>
      </section>
    </div>
  )
}

function summarize(runs: ScoredRun[]): Summary[] {
  const byModel = new Map<string, ScoredRun[]>()
  runs.forEach((run) => {
    const bucket = byModel.get(run.model)
    if (bucket) bucket.push(run)
    else byModel.set(run.model, [run])
  })

  return [...byModel.entries()]
    .map(([model, rows]) => {
      const rtfs = rows.map((r) => r.rtf).filter((v): v is number => v != null)
      const wers = rows.map((r) => r.wer?.wer).filter((v): v is number => v != null)
      return {
        model,
        runs: rows.length,
        meanRtf: rtfs.length ? rtfs.reduce((a, b) => a + b, 0) / rtfs.length : null,
        meanWer: wers.length ? wers.reduce((a, b) => a + b, 0) / wers.length : null,
        scored: wers.length,
      }
    })
    .sort((a, b) => a.model.localeCompare(b.model))
}


interface ExpectedCellProps {
  run: ScoredRun
  editing: boolean
  saving: boolean
  error: string | null
  onEdit: () => void
  onCancel: () => void
  onSave: (runId: number, value: string) => void
}

/**
 * The "what you meant" half of the row.
 *
 * Read mode shows the stored ground truth (or a prompt to add one); edit mode
 * swaps in a textarea pre-filled with it. The draft is local state so typing
 * never re-renders the whole table, and it is seeded from the run each time the
 * editor opens rather than being held for closed rows.
 */
function ExpectedCell({
  run,
  editing,
  saving,
  error,
  onEdit,
  onCancel,
  onSave,
}: ExpectedCellProps) {
  const [draft, setDraft] = useState(run.expected_text ?? '')
  const textarea = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    if (editing) {
      setDraft(run.expected_text ?? '')
      textarea.current?.focus()
    }
  }, [editing, run.expected_text])

  if (!editing) {
    return (
      <div className="expected-cell">
        {run.expected_text ? (
          <span className="expected-value">{run.expected_text}</span>
        ) : (
          <span className="faint">no ground truth</span>
        )}
        <button
          className="btn tiny"
          onClick={onEdit}
          title="Edit the expected text for this run"
        >
          {run.expected_text ? 'Edit' : 'Add'}
        </button>
      </div>
    )
  }

  return (
    <div className="expected-editor">
      <textarea
        ref={textarea}
        value={draft}
        disabled={saving}
        rows={3}
        placeholder="What was actually said. Leave empty to clear the label."
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          // Enter inserts a newline; Ctrl/Cmd+Enter saves, Escape discards.
          if (event.key === 'Escape') onCancel()
          if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
            onSave(run.id, draft)
          }
        }}
      />
      <div className="row" style={{ gap: 6 }}>
        <button
          className="btn primary tiny"
          onClick={() => onSave(run.id, draft)}
          disabled={saving}
        >
          {saving ? 'Saving…' : 'Save'}
        </button>
        <button className="btn tiny" onClick={onCancel} disabled={saving}>
          Cancel
        </button>
        <span className="faint">⌘/Ctrl+Enter</span>
      </div>
      {error && <div className="notice error">{error}</div>}
    </div>
  )
}
