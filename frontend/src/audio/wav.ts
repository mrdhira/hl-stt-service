/**
 * WAV encoding.
 *
 * The backend accepts WAV only (stdlib `wave`, no ffmpeg), but MediaRecorder
 * hands us webm/opus. So: decode with the WebAudio decoder the browser already
 * has, remix/resample through an OfflineAudioContext, then write a canonical
 * 16-bit mono RIFF header ourselves. No ffmpeg.wasm, no extra megabytes.
 */

/** Target rate: what all three models want, and it keeps uploads small. */
export const TARGET_SAMPLE_RATE = 16000

export interface WavClip {
  blob: Blob
  sampleRate: number
  durationMs: number
}

/** Encode mono float samples in [-1, 1] as a 16-bit PCM WAV. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample)
  const view = new DataView(buffer)

  const writeAscii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i))
  }

  const dataBytes = samples.length * bytesPerSample
  writeAscii(0, 'RIFF')
  view.setUint32(4, 36 + dataBytes, true)
  writeAscii(8, 'WAVE')
  writeAscii(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM fmt chunk size
  view.setUint16(20, 1, true) // format = PCM
  view.setUint16(22, 1, true) // channels = mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * bytesPerSample, true) // byte rate
  view.setUint16(32, bytesPerSample, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  writeAscii(36, 'data')
  view.setUint32(40, dataBytes, true)

  floatToPcm16(samples, new DataView(buffer, 44))
  return new Blob([buffer], { type: 'audio/wav' })
}

/** Clamp + scale float samples into a little-endian int16 view. */
export function floatToPcm16(samples: Float32Array, out: DataView): void {
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]))
    // Asymmetric on purpose: int16 range is [-32768, 32767].
    out.setInt16(i * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true)
  }
}

/** Standalone int16 buffer, for the websocket path. */
export function floatToPcm16Buffer(samples: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(samples.length * 2)
  floatToPcm16(samples, new DataView(buffer))
  return buffer
}

/**
 * Turn whatever MediaRecorder produced into a 16 kHz mono WAV.
 *
 * `decodeAudioData` handles the container/codec; `OfflineAudioContext` does the
 * downmix and sample-rate conversion in one render pass.
 */
export async function blobToWav(
  input: Blob,
  targetRate: number = TARGET_SAMPLE_RATE,
): Promise<WavClip> {
  const bytes = await input.arrayBuffer()
  if (bytes.byteLength === 0) throw new Error('recording is empty')

  // A plain AudioContext just for decoding; closed straight after.
  const decodeCtx = new AudioContext()
  let decoded: AudioBuffer
  try {
    decoded = await decodeCtx.decodeAudioData(bytes)
  } catch (cause) {
    throw new Error(
      `browser could not decode the recording (${input.type || 'unknown type'})`,
      { cause },
    )
  } finally {
    void decodeCtx.close()
  }

  const mono = await resampleToMono(decoded, targetRate)
  return {
    blob: encodeWav(mono, targetRate),
    sampleRate: targetRate,
    durationMs: (mono.length / targetRate) * 1000,
  }
}

/** Downmix to one channel and resample, via an offline render. */
export async function resampleToMono(
  source: AudioBuffer,
  targetRate: number,
): Promise<Float32Array> {
  const frames = Math.max(1, Math.ceil((source.duration * targetRate) || 1))
  const offline = new OfflineAudioContext(1, frames, targetRate)

  const node = offline.createBufferSource()
  node.buffer = source
  // Multi-channel sources are averaged down by the 1-channel destination.
  node.connect(offline.destination)
  node.start()

  const rendered = await offline.startRendering()
  // Copy out of the AudioBuffer: its internal storage is not ours to keep.
  const out = new Float32Array(rendered.length)
  rendered.copyFromChannel(out, 0)
  return out
}

/** Best-effort MediaRecorder mime type for this browser. */
export function pickRecorderMimeType(): string | undefined {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ]
  if (typeof MediaRecorder === 'undefined') return undefined
  return candidates.find((type) => MediaRecorder.isTypeSupported(type))
}
