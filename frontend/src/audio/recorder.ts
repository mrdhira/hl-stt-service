/** getUserMedia + MediaRecorder, wrapped so the tab component stays readable. */

import { pickRecorderMimeType } from './wav'

export interface Recording {
  blob: Blob
  mimeType: string
}

export class MicRecorder {
  private stream: MediaStream | null = null
  private recorder: MediaRecorder | null = null
  private chunks: Blob[] = []

  get active(): boolean {
    return this.recorder?.state === 'recording'
  }

  async start(): Promise<void> {
    if (this.active) return
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

    const mimeType = pickRecorderMimeType()
    this.chunks = []
    this.recorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined)
    this.recorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.chunks.push(event.data)
    }
    this.recorder.start()
  }

  /** Stop and resolve with everything captured. Always releases the mic. */
  stop(): Promise<Recording> {
    const recorder = this.recorder
    if (!recorder || recorder.state === 'inactive') {
      this.release()
      return Promise.reject(new Error('not recording'))
    }

    return new Promise<Recording>((resolve, reject) => {
      recorder.onstop = () => {
        const mimeType = recorder.mimeType || this.chunks[0]?.type || 'audio/webm'
        const blob = new Blob(this.chunks, { type: mimeType })
        this.release()
        if (blob.size === 0) reject(new Error('recording is empty'))
        else resolve({ blob, mimeType })
      }
      recorder.onerror = (event) => {
        this.release()
        reject(new Error(`recorder failed: ${String(event)}`))
      }
      recorder.stop()
    })
  }

  /** Drop the recorder and stop every mic track (turns the browser dot off). */
  release(): void {
    this.stream?.getTracks().forEach((track) => track.stop())
    this.stream = null
    this.recorder = null
  }
}
