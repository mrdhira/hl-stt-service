import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, postTranscribe } from '../api/client'
import type { ModelInfo, TranscribeResult } from '../api/types'
import { MicRecorder } from '../audio/recorder'
import { TARGET_SAMPLE_RATE, blobToWav } from '../audio/wav'
import type { WavClip } from '../audio/wav'
import { ModelPicker } from '../components/ModelPicker'
import { MetricGrid } from '../components/Metrics'
import { Transcript } from '../components/Transcript'
import { count, ms, percent, rtf, rtfTone, werTone } from '../lib/format'
import { wordErrorRate } from '../lib/wer'

interface Props {
  models: ModelInfo[]
  model: string
  onModelChange: (key: string) => void
  onRunSaved: () => void
}

type Phase = 'idle' | 'recording' | 'converting' | 'sending'

export function RecordSend({ models, model, onModelChange, onRunSaved }: Props) {
  const [phase, setPhase] = useState<Phase>('idle')
  const [clip, setClip] = useState<WavClip | null>(null)
  const [clipUrl, setClipUrl] = useState<string | null>(null)
  const [expectedText, setExpectedText] = useState('')
  const [result, setResult] = useState<TranscribeResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)

  const recorder = useRef<MicRecorder | null>(null)
  if (recorder.current === null) recorder.current = new MicRecorder()

  // Recording timer.
  useEffect(() => {
    if (phase !== 'recording') return
    const startedAt = performance.now()
    setElapsed(0)
    const timer = window.setInterval(() => setElapsed(performance.now() - startedAt), 100)
    return () => window.clearInterval(timer)
  }, [phase])

  // Release the mic and the object URL if the tab unmounts mid-take.
  useEffect(() => {
    const mic = recorder.current
    return () => mic?.release()
  }, [])
  useEffect(() => {
    return () => {
      if (clipUrl) URL.revokeObjectURL(clipUrl)
    }
  }, [clipUrl])

  const replaceClip = useCallback((next: WavClip | null) => {
    setClip(next)
    setClipUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous)
      return next ? URL.createObjectURL(next.blob) : null
    })
  }, [])

  const startRecording = async () => {
    setError(null)
    setResult(null)
    replaceClip(null)
    try {
      await recorder.current!.start()
      setPhase('recording')
    } catch (cause) {
      setPhase('idle')
      setError(cause instanceof Error ? cause.message : 'could not start the microphone')
    }
  }

  const stopRecording = async () => {
    setPhase('converting')
    try {
      const recording = await recorder.current!.stop()
      // MediaRecorder gives webm/opus; the backend takes WAV only.
      const wav = await blobToWav(recording.blob, TARGET_SAMPLE_RATE)
      replaceClip(wav)
      setPhase('idle')
    } catch (cause) {
      setPhase('idle')
      setError(cause instanceof Error ? cause.message : 'could not process the recording')
    }
  }

  const send = async () => {
    if (!clip || !model) return
    setPhase('sending')
    setError(null)
    setResult(null)
    try {
      const response = await postTranscribe(model, clip.blob, expectedText)
      setResult(response)
      onRunSaved()
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setPhase('idle')
    }
  }

  const selected = models.find((m) => m.key === model)
  const busy = phase !== 'idle'
  // WER against what the user typed, computed client-side; the backend just
  // stores the ground truth alongside the run.
  const wer = result ? wordErrorRate(result.expected_text ?? expectedText, result.text) : null

  return (
    <div className="stack">
      <section className="panel">
        <h2>Record</h2>
        <div className="row" style={{ alignItems: 'flex-start' }}>
          <ModelPicker
            models={models}
            value={model}
            onChange={onModelChange}
            disabled={busy}
          />

          <div className="stack grow" style={{ gap: 10 }}>
            <div className="row">
              {phase === 'recording' ? (
                <button className="btn danger" onClick={stopRecording}>
                  <span className="dot pulse" style={{ display: 'inline-block', marginRight: 6 }} />
                  Stop · {(elapsed / 1000).toFixed(1)}s
                </button>
              ) : (
                <button className="btn" onClick={startRecording} disabled={busy}>
                  Record
                </button>
              )}

              <button
                className="btn primary"
                onClick={send}
                disabled={!clip || !selected?.available || busy}
              >
                {phase === 'sending' ? 'Transcribing…' : 'Transcribe'}
              </button>

              {phase === 'converting' && <span className="faint">Converting to 16 kHz WAV…</span>}

              {clip && phase === 'idle' && (
                <span className="badge ok">
                  <span className="dot" />
                  {(clip.durationMs / 1000).toFixed(1)}s · {clip.sampleRate / 1000} kHz mono WAV ·{' '}
                  {(clip.blob.size / 1024).toFixed(0)} KB
                </span>
              )}
            </div>

            {clipUrl && (
              <audio controls src={clipUrl} style={{ width: '100%', maxWidth: 420 }} />
            )}
          </div>
        </div>
      </section>

      <section className="panel">
        <h2>Test case</h2>
        <label className="field">
          Expected text — what was actually said. Saved with the run so Reports can
          score WER. Leave empty for an untagged run.
          <textarea
            value={expectedText}
            onChange={(event) => setExpectedText(event.target.value)}
            placeholder="e.g. the quick brown fox / 今日はいい天気ですね"
            disabled={busy}
          />
        </label>
      </section>

      {error && <div className="notice error">{error}</div>}

      {result && (
        <section className="panel">
          <h2>Result · {result.model}</h2>
          <div className="stack">
            <Transcript text={result.text} placeholder="Model returned no text." />
            <MetricGrid
              items={[
                { label: 'RTF', value: rtf(result.rtf), tone: rtfTone(result.rtf), hint: '<1 = faster than realtime' },
                { label: 'Audio', value: ms(result.audio_ms) },
                { label: 'Processing', value: ms(result.processing_ms) },
                { label: 'Words', value: count(result.words) },
                { label: 'Chars', value: count(result.chars) },
                {
                  label: 'WER',
                  value: wer ? percent(wer.wer) : '—',
                  tone: werTone(wer?.wer),
                  hint: wer ? `${wer.substitutions}S ${wer.deletions}D ${wer.insertions}I / ${wer.referenceLength}` : 'no expected text',
                },
              ]}
            />
            <div className="faint">
              run #{result.run_id ?? '—'} · {result.sample_rate} Hz · hash{' '}
              <span className="mono">{result.text_hash.slice(0, 12)}</span>
              {result.expected_text && ' · saved as a test case'}
            </div>
          </div>
        </section>
      )}
    </div>
  )
}
