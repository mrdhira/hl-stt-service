/**
 * Live 16-bit PCM capture for WS /stream.
 *
 * Prefers an AudioWorklet (runs on the audio thread, no main-thread jank) and
 * falls back to ScriptProcessorNode where AudioWorklet is unavailable. The
 * worklet source is inlined and loaded from a blob URL so there is no extra
 * asset to serve or path to get wrong behind a reverse proxy.
 */

import { floatToPcm16Buffer } from './wav'

const WORKLET_SOURCE = `
class PcmTap extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0]
    if (channel && channel.length > 0) {
      // Copy: the render quantum buffer is reused between calls.
      this.port.postMessage(new Float32Array(channel))
    }
    return true
  }
}
registerProcessor('pcm-tap', PcmTap)
`

export interface PcmCaptureOptions {
  /** Requested capture rate. The real rate is reported by `start()`. */
  sampleRate: number
  /** Called with each int16 chunk, ready to hand to WebSocket.send. */
  onChunk: (pcm: ArrayBuffer) => void
}

export class PcmCapture {
  private context: AudioContext | null = null
  private stream: MediaStream | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private node: AudioWorkletNode | ScriptProcessorNode | null = null
  private workletUrl: string | null = null
  private readonly options: PcmCaptureOptions

  constructor(options: PcmCaptureOptions) {
    this.options = options
  }

  /** Start capturing. Returns the sample rate actually in use. */
  async start(): Promise<number> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('this browser has no microphone API (needs HTTPS or localhost)')
    }

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    })

    // Browsers may refuse the requested rate and pick the device's own; we read
    // back context.sampleRate and tell the backend what it really is.
    this.context = new AudioContext({ sampleRate: this.options.sampleRate })
    if (this.context.state === 'suspended') await this.context.resume()

    this.source = this.context.createMediaStreamSource(this.stream)

    if (this.context.audioWorklet) {
      await this.startWorklet()
    } else {
      this.startScriptProcessor()
    }

    return this.context.sampleRate
  }

  private async startWorklet(): Promise<void> {
    const context = this.context!
    this.workletUrl = URL.createObjectURL(
      new Blob([WORKLET_SOURCE], { type: 'application/javascript' }),
    )
    await context.audioWorklet.addModule(this.workletUrl)

    const node = new AudioWorkletNode(context, 'pcm-tap', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      channelCount: 1,
    })
    node.port.onmessage = (event: MessageEvent<Float32Array>) => {
      this.options.onChunk(floatToPcm16Buffer(event.data))
    }
    this.source!.connect(node)
    this.node = node
  }

  private startScriptProcessor(): void {
    const context = this.context!
    // Deprecated but still the only fallback; 4096 frames ~= 256 ms at 16 kHz.
    const node = context.createScriptProcessor(4096, 1, 1)
    node.onaudioprocess = (event) => {
      const channel = event.inputBuffer.getChannelData(0)
      this.options.onChunk(floatToPcm16Buffer(new Float32Array(channel)))
    }
    this.source!.connect(node)
    // ScriptProcessorNode only fires while connected to a destination; a muted
    // gain node keeps it running without echoing the mic to the speakers.
    const mute = context.createGain()
    mute.gain.value = 0
    node.connect(mute)
    mute.connect(context.destination)
    this.node = node
  }

  /** Stop capturing and release the mic. Safe to call more than once. */
  stop(): void {
    if (this.node) {
      this.node.disconnect()
      if ('port' in this.node) this.node.port.onmessage = null
      else this.node.onaudioprocess = null
    }
    this.source?.disconnect()
    this.stream?.getTracks().forEach((track) => track.stop())
    void this.context?.close()
    if (this.workletUrl) URL.revokeObjectURL(this.workletUrl)

    this.node = null
    this.source = null
    this.stream = null
    this.context = null
    this.workletUrl = null
  }
}
