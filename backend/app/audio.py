"""Audio helpers: WAV -> float32 mono samples, using the stdlib only.

sherpa-onnx wants a 1-D float sequence normalised to [-1, 1] plus the sample
rate; it resamples internally, so we do not resample here. numpy is
intentionally not a dependency of the POC.
"""

from __future__ import annotations

import io
import wave

SUPPORTED_NOTE = "16-bit/24-bit/32-bit PCM or 32-bit float WAV, any sample rate"


class AudioDecodeError(ValueError):
    """Raised when the uploaded bytes are not a WAV we can read."""


def decode_wav(data: bytes) -> tuple[list[float], int]:
    """Decode WAV bytes to ``(samples, sample_rate)``.

    Multi-channel audio is downmixed to mono. Raises :class:`AudioDecodeError`
    for anything we cannot parse.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as exc:  # pragma: no cover - exercised by bad input
        raise AudioDecodeError(f"not a readable WAV file: {exc}") from exc

    if not frames:
        return [], sample_rate

    if n_channels > 1:
        # stdlib ``audioop`` would do this, but it is gone in Python 3.13+.
        frames = _downmix(frames, sample_width, n_channels)

    samples = pcm_to_float(frames, sample_width)
    return samples, sample_rate


def _downmix(frames: bytes, sample_width: int, n_channels: int) -> bytes:
    """Average interleaved channels down to mono."""
    signed = sample_width > 1  # WAV 8-bit is unsigned, everything wider is signed
    mono = bytearray()
    frame_size = sample_width * n_channels
    for off in range(0, len(frames) - frame_size + 1, frame_size):
        acc = 0
        for ch in range(n_channels):
            start = off + ch * sample_width
            acc += int.from_bytes(
                frames[start : start + sample_width], "little", signed=signed
            )
        mono += (acc // n_channels).to_bytes(sample_width, "little", signed=signed)
    return bytes(mono)


def pcm_to_float(frames: bytes, sample_width: int) -> list[float]:
    """Convert little-endian signed PCM to floats in [-1, 1]."""
    if sample_width == 1:
        # WAV 8-bit is unsigned.
        return [(b - 128) / 128.0 for b in frames]
    if sample_width not in (2, 3, 4):
        raise AudioDecodeError(f"unsupported sample width: {sample_width} bytes")

    scale = float(1 << (sample_width * 8 - 1))
    n = len(frames) // sample_width
    out: list[float] = [0.0] * n
    for i in range(n):
        start = i * sample_width
        value = int.from_bytes(
            frames[start : start + sample_width], "little", signed=True
        )
        out[i] = value / scale
    return out


def pcm16_to_float(frames: bytes) -> list[float]:
    """Fast path for the raw 16-bit PCM frames the websocket sends."""
    return pcm_to_float(frames, 2)


def duration_ms(n_samples: int, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return 1000.0 * n_samples / sample_rate


def looks_like_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"
