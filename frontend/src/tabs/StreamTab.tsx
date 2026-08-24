import { useCallback, useEffect, useRef, useState } from 'react'

import { STREAM_SAMPLE_RATE, wsUrl } from '../api/config'
import type { ModelInfo, StreamFinal, StreamMessage, StreamPartial } from '../api/types'
import { PcmCapture } from '../audio/pcmCapture'
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

type Phase = 'idle' | 'connecting' | 'live' | 'finishing'

export function StreamTab({ models, model, onModelChange, onRunSaved }: Props) {
  const [phase, setPhase] = useState<Phase>('idle')
  const [partial, setPartial] = useState<StreamPartial | null>(null)
  const [partialCount, setPartialCount] = useState(0)
  const [final, setFinal] = useState<StreamFinal | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [expectedText, setExpectedText] = useState('')
  const [note, setNote] = useState<string | null>(null)

  const socket = useRef<WebSocket | null>(null)
  const capture = useRef<PcmCapture | null>(null)
  // Read inside the socket callbacks, which close over the first render.
  const expectedRef = useRef('')
  expectedRef.current = expectedText

  const teardown = useCallback(() => {
    capture.current?.stop()
    capture.current = null
    const ws = socket.current
    socket.current = null
    if (ws && ws.readyState <= WebSocket.OPEN) ws.close()
  }, [])

  useEffect(() => teardown, [teardown])

  const start = async () => {
    setError(null)
    setFinal(null)
    setPartial(null)
    setPartialCount(0)
    setNote(null)
    setPhase('connecting')

    // Open the mic first: the sample rate the browser actually gives us is what
    // the backend needs to be told, and asking after connecting would race.
    let actualRate: number
    const pcm = new PcmCapture({
      sampleRate: STREAM_SAMPLE_RATE,
      onChunk: (chunk) => {
        const ws = socket.current
        if (ws?.readyState === WebSocket.OPEN) ws.send(chunk)
      },
    })
    try {
      actualRate = await pcm.start()
    } catch (cause) {
      pcm.stop()
      setPhase('idle')
      setError(cause instanceof Error ? cause.message : 'could not start the microphone')
      return
    }
    capture.current = pcm

    const ws = new WebSocket(
      wsUrl(`/stream?model=${encodeURIComponent(model)}&sample_rate=${actualRate}`),
    )
    ws.binaryType = 'arraybuffer'
    socket.current = ws

    ws.onmessage = (event) => {
      let message: StreamMessage
      try {
        message = JSON.parse(String(event.data)) as StreamMessage
      } catch {
        return
      }

      switch (message.type) {
        case 'ready':
          setPhase('live')
          if (!message.supports_true_streaming) setNote(message.note)
          break
        case 'partial':
          setPartial(message)
          setPartialCount((n) => n + 1)
          break
        case 'final':
          setFinal(message)
          setPartial(null)
          setPhase('idle')
          teardown()
          onRunSaved()
          break
        case 'error':
          setError(message.detail)
          setPhase('idle')
          teardown()
          break
        case 'reset':
          break
      }
    }

    ws.onerror = () => {
      setError('websocket connection failed — is the backend running?')
      setPhase('idle')
      teardown()
    }

    ws.onclose = () => {
      // A close before the final means the run never landed.
      setPhase((current) => (current === 'idle' ? current : 'idle'))
      capture.current?.stop()
      capture.current = null
    }
  }

  /** Stop the mic, then ask for the final decode. */
  const finish = () => {
    const ws = socket.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    setPhase('finishing')
    // Stop capturing before eof so no frame arrives after it.
    capture.current?.stop()
    capture.current = null
    ws.send(JSON.stringify({ type: 'eof', expected_text: expectedRef.current.trim() }))
  }

  const cancel = () => {
    teardown()
    setPhase('idle')
    setPartial(null)
  }

  const selected = models.find((m) => m.key === model)
  const busy = phase !== 'idle'
  const wer = final ? wordErrorRate(final.expected_text ?? expectedText, final.text) : null

  return (
    <div className="stack">
      <section className="panel">
        <h2>Stream</h2>
        <div className="row" style={{ alignItems: 'center' }}>
          <ModelPicker
            models={models}
            value={model}
            onChange={onModelChange}
            disabled={busy}
            requireStreaming
          />

          <div className="stack grow" style={{ gap: 10 }}>
            <div className="row">
              {phase === 'idle' ? (
                <button
                  className="btn primary"
                  onClick={start}
                  disabled={!selected?.available}
                >
                  Start streaming
                </button>
              ) : (
                <>
                  <button className="btn danger" onClick={finish} disabled={phase !== 'live'}>
                    {phase === 'finishing' ? 'Decoding final…' : 'Stop & finalize'}
                  </button>
                  <button className="btn" onClick={cancel}>
                    Cancel
                  </button>
                </>
              )}

              {phase === 'connecting' && <span className="faint">Connecting…</span>}
              {phase === 'live' && (
                <span className="badge live">
                  <span className="dot pulse" />
                  live · {partialCount} partial{partialCount === 1 ? '' : 's'}
                </span>
              )}
            </div>

            {note && <div className="notice">{note}</div>}
          </div>
        </div>
      </section>

      <section className="panel">
        <h2>Test case</h2>
        <label className="field">
          Expected text — sent with the final frame and stored on the run for WER.
          <textarea
            value={expectedText}
            onChange={(event) => setExpectedText(event.target.value)}
            placeholder="What you are about to say"
          />
        </label>
      </section>

      {error && <div className="notice error">{error}</div>}

      <section className="panel">
        <h2>{final ? 'Final' : 'Live transcript'}</h2>
        <div className="stack">
          {final ? (
            <Transcript text={final.text} placeholder="Model returned no text." />
          ) : (
            <Transcript
              text={partial?.text ?? ''}
              partial
              placeholder={
                phase === 'live'
                  ? 'Listening… the first partial lands once enough audio has arrived.'
                  : 'Start streaming to see partials here.'
              }
            />
          )}

          {partial && !final && (
            <div className="faint">
              partial at {ms(partial.elapsed_ms)} · {ms(partial.audio_ms)} audio buffered ·
              decode {ms(partial.processing_ms)}
            </div>
          )}

          {final && (
            <>
              <MetricGrid
                items={[
                  {
                    label: 'First partial',
                    value: ms(final.latency_partial_ms),
                    hint: 'from first audio',
                  },
                  {
                    label: 'Final latency',
                    value: ms(final.latency_final_ms),
                    hint: 'from end of audio',
                  },
                  { label: 'Audio', value: ms(final.audio_ms) },
                  {
                    label: 'RTF',
                    value: rtf(final.rtf),
                    tone: rtfTone(final.rtf),
                    hint: 'sum of all decodes',
                  },
                  { label: 'Words', value: count(final.words) },
                  { label: 'Chars', value: count(final.chars) },
                  {
                    label: 'WER',
                    value: wer ? percent(wer.wer) : '—',
                    tone: werTone(wer?.wer),
                    hint: wer
                      ? `${wer.substitutions}S ${wer.deletions}D ${wer.insertions}I / ${wer.referenceLength}`
                      : 'no expected text',
                  },
                ]}
              />
              <div className="faint">
                run #{final.run_id ?? '—'} · {final.partials_emitted} partials ·{' '}
                final decode {ms(final.final_decode_ms)} · {final.sample_rate} Hz
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  )
}
