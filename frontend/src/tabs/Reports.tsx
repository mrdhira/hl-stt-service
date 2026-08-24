import { useMemo, useState } from 'react'

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
                  <th>Text</th>
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
                      {run.expected_text && (
                        <span className="expected">expected: {run.expected_text}</span>
                      )}
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
