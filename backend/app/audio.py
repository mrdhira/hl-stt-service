"""Audio helpers: WAV -> float32 mono samples, using the stdlib only.

sherpa-onnx wants a 1-D float sequence normalised to [-1, 1] plus the sample
rate; it resamples internally, so we do not resample here. numpy is
intentionally not a dependency of the POC.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import wave

from .config import settings

SUPPORTED_NOTE = "16-bit/24-bit/32-bit PCM or 32-bit float WAV, any sample rate"


class AudioDecodeError(ValueError):
    """Raised when the uploaded bytes are not audio we can read."""


class EmptyAudioError(AudioDecodeError):
    """Raised for a zero-byte upload.

    A subclass so callers that only care about "could not decode" keep working,
    while the routes can still answer 400 (you sent nothing) rather than 415
    (I do not understand this format).
    """


class AudioTooLongError(AudioDecodeError):
    """Raised when decoded audio exceeds ``STT_MAX_AUDIO_SECONDS``.

    Separate from a plain decode failure because the two mean different things
    to a caller: one is "I cannot read this", the other is "I could, but it is
    bigger than I am willing to hold in memory".
    """


def check_duration(n_samples: int, sample_rate: int, max_seconds: float | None = None) -> None:
    """Reject over-long audio *before* it is expanded into Python floats.

    This is the load-bearing half of the DoS guard. PCM bytes cost 2 per sample;
    the float list that sherpa-onnx needs costs ~32 (a 64-bit float plus its
    pointer in the list), so the conversion is a ~16x amplification on top of
    whatever the codec already expanded. Checking the sample count first keeps
    the worst case bounded no matter how well the upload compressed.
    """
    limit = settings.max_audio_seconds if max_seconds is None else max_seconds
    if limit <= 0 or sample_rate <= 0:
        return
    seconds = n_samples / sample_rate
    if seconds > limit:
        raise AudioTooLongError(
            f"audio is {seconds:.1f}s, which exceeds the {limit:.0f}s limit"
        )


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
            # The header declares the length, so the limit is enforced before a
            # single frame is read rather than after.
            check_duration(wf.getnframes(), sample_rate)
            frames = wf.readframes(wf.getnframes())
    except AudioTooLongError:
        raise
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


# ---------------------------------------------------------------------------
# Compressed audio (Telegram voice notes, browser recordings)
# ---------------------------------------------------------------------------
# WAV stays on the stdlib path above — zero dependencies, no subprocess. Every
# other container is handed to one of two optional backends:
#
#   soundfile  in-process (libsndfile). Covers OGG/Opus, OGG/Vorbis, FLAC, MP3.
#              This is the Telegram voice-note path, so it is tried first.
#   ffmpeg     subprocess. Covers everything else, notably webm/opus, which
#              libsndfile cannot read (it has no Matroska demuxer).
#
# Both are optional: with neither installed, WAV still works and anything else
# raises a clear AudioDecodeError naming what to install.

try:  # pragma: no cover - depends on the host
    import soundfile as _soundfile

    SOUNDFILE_AVAILABLE = True
except Exception:  # pragma: no cover
    _soundfile = None  # type: ignore[assignment]
    SOUNDFILE_AVAILABLE = False


def ffmpeg_path() -> str | None:
    """Path to the ffmpeg binary, or None when it is not installed."""
    return shutil.which("ffmpeg")


# Magic-number sniffing. Content-Type from a multipart upload is whatever the
# client felt like sending, so the bytes are the only trustworthy source.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"OggS", "ogg"),
    (b"\x1a\x45\xdf\xa3", "webm"),  # EBML — webm or matroska
    (b"fLaC", "flac"),
    (b"ID3", "mp3"),
)

#: Formats the /asr endpoint advertises. WAV is always available; the rest need
#: soundfile or ffmpeg.
SUPPORTED_UPLOAD_NOTE = "WAV, OGG/Opus, OGG/Vorbis, WebM/Opus, FLAC, MP3, MP4/M4A"


def sniff_format(data: bytes) -> str:
    """Best-effort container name from the leading bytes.

    Returns one of wav/ogg/webm/flac/mp3/mp4, or "bin" when nothing matches.
    """
    if looks_like_wav(data):
        return "wav"
    for magic, name in _MAGIC:
        if data.startswith(magic):
            return name
    # ISO-BMFF (mp4/m4a) puts a size field before the 'ftyp' box.
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "mp4"
    # Bare MPEG audio frame sync, for mp3 without an ID3 header.
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"
    return "bin"


def decode_audio(data: bytes) -> tuple[list[float], int]:
    """Decode any supported container to ``(mono float samples, sample_rate)``.

    Samples are normalised to [-1, 1], which is what sherpa-onnx wants. No
    resampling happens here — sherpa resamples internally, so the true rate is
    returned alongside.
    """
    if not data:
        raise EmptyAudioError("empty upload")

    container = sniff_format(data)
    if container == "wav":
        return decode_wav(data)

    errors: list[str] = []

    # libsndfile first: in-process, and it covers the Telegram OGG/Opus case.
    # AudioTooLongError propagates immediately everywhere below: audio over the
    # duration cap is over it for every decoder, so falling through only burns
    # CPU and would report the failure as "unsupported format" (415) instead of
    # "too long" (413).
    if SOUNDFILE_AVAILABLE and container in ("ogg", "flac", "mp3"):
        try:
            return _decode_with_soundfile(data)
        except AudioTooLongError:
            raise
        except Exception as exc:  # fall through to ffmpeg
            errors.append(f"soundfile: {exc}")

    if ffmpeg_path():
        try:
            return _decode_with_ffmpeg(data, container)
        except AudioTooLongError:
            raise
        except AudioDecodeError as exc:
            errors.append(str(exc))
    else:
        errors.append("ffmpeg: not installed")

    # Last resort: libsndfile on a container it does not advertise. Cheap to try
    # and it occasionally succeeds on mislabelled uploads.
    if SOUNDFILE_AVAILABLE and container not in ("ogg", "flac", "mp3"):
        try:
            return _decode_with_soundfile(data)
        except AudioTooLongError:
            raise
        except Exception as exc:
            errors.append(f"soundfile: {exc}")

    detail = "; ".join(errors) if errors else "no decoder available"
    raise AudioDecodeError(
        f"could not decode {container!r} audio ({detail}). "
        f"Supported: {SUPPORTED_UPLOAD_NOTE}. "
        "Install ffmpeg for webm/mp4, or the `soundfile` package for ogg/flac/mp3."
    )


def _decode_with_soundfile(data: bytes) -> tuple[list[float], int]:
    """Decode via libsndfile, downmixing to mono."""
    with io.BytesIO(data) as handle:
        # SoundFile exposes the frame count from the header, so the length is
        # checked before any samples are materialised.
        with _soundfile.SoundFile(handle) as info:
            check_duration(info.frames, info.samplerate)
        handle.seek(0)
        frames, sample_rate = _soundfile.read(
            handle, dtype="float32", always_2d=True
        )
    if frames.shape[0] == 0:
        raise AudioDecodeError("audio contains no samples")
    mono = frames.mean(axis=1) if frames.shape[1] > 1 else frames[:, 0]
    return mono.tolist(), int(sample_rate)


#: Rate ffmpeg decodes to. All three models want 16 kHz, and asking ffmpeg to
#: resample is cheaper and better than making sherpa do it.
FFMPEG_SAMPLE_RATE = 16000

#: Guard against a malformed or hostile upload wedging the request.
FFMPEG_TIMEOUT_S = 120


#: Sniffed container -> the ffmpeg demuxer to force with `-f`. Pinning this
#: stops ffmpeg picking a demuxer from hostile bytes: without it, content that
#: merely *looks* like a playlist can steer the input layer somewhere we never
#: intended. Anything not listed here is decoded without a pinned demuxer, but
#: still under the protocol whitelist below.
_FFMPEG_DEMUXER = {
    "ogg": "ogg",
    "webm": "matroska",
    "mp4": "mov,mp4,m4a,3gp,3g2,mj2",
    "flac": "flac",
    "mp3": "mp3",
    "wav": "wav",
}


def _decode_with_ffmpeg(data: bytes, container: str) -> tuple[list[float], int]:
    """Decode via an ffmpeg subprocess to 16 kHz mono s16le.

    The input goes through a temp file rather than stdin: the Matroska demuxer
    seeks, and piping webm to ffmpeg fails for exactly that reason.

    Hardened on three axes, because the bytes are attacker-controlled:
      * `-protocol_whitelist file` — the input is a local temp file and nothing
        else, so a container that references http/hls/concat targets cannot make
        ffmpeg fetch them.
      * `-f <demuxer>` — the demuxer is chosen from our own magic-byte sniff,
        not from the file's content.
      * `-t` — output is bounded, so a decode bomb cannot fill memory before the
        duration check downstream gets a chance to reject it.
    """
    binary = ffmpeg_path()
    if not binary:  # pragma: no cover - guarded by the caller
        raise AudioDecodeError("ffmpeg: not installed")

    suffix = f".{container}" if container != "bin" else ""
    with tempfile.NamedTemporaryFile(suffix=suffix) as source:
        source.write(data)
        source.flush()

        # One second past the cap: enough for check_duration() to see the
        # overrun and reject, without decoding an unbounded amount first.
        max_output_s = settings.max_audio_seconds + 1
        demuxer = _FFMPEG_DEMUXER.get(container)

        command = [
            binary,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-protocol_whitelist", "file",
        ]
        if demuxer:
            command += ["-f", demuxer]
        command += [
            "-i", source.name,
            "-vn",                      # ignore album art / video tracks
            "-map", "a:0",              # first audio stream only
            "-ac", "1",
            "-ar", str(FFMPEG_SAMPLE_RATE),
            "-t", f"{max_output_s:.3f}",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                timeout=FFMPEG_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioDecodeError(
                f"ffmpeg: timed out after {FFMPEG_TIMEOUT_S}s"
            ) from exc

    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = message[-1] if message else f"exit {result.returncode}"
        raise AudioDecodeError(f"ffmpeg: {tail}")
    if not result.stdout:
        raise AudioDecodeError("ffmpeg: produced no audio")

    # Two bytes per sample. Checked before pcm16_to_float, which is where the
    # ~16x float amplification would otherwise happen.
    check_duration(len(result.stdout) // 2, FFMPEG_SAMPLE_RATE)
    return pcm16_to_float(result.stdout), FFMPEG_SAMPLE_RATE


#: Container name -> file extension used when archiving the clip to the dataset.
_DATASET_EXTENSION = {
    "wav": "wav",
    "ogg": "ogg",
    "webm": "webm",
    "flac": "flac",
    "mp3": "mp3",
    "mp4": "m4a",
    "bin": "bin",
}


def dataset_extension(data: bytes) -> str:
    """Extension for archiving these bytes, decided solely by magic bytes.

    The upload's filename is deliberately ignored. It is attacker-controlled,
    and it used to be the fallback when sniffing failed — which let a caller
    name the archived file `.php`, `.sh` or `.htm`. The dataset directory is a
    host bind mount, so a caller-chosen extension there is a foothold worth
    denying even though nothing serves that directory today.

    Unrecognised content gets `bin`, always. The container is recoverable from
    the bytes if it is ever needed.
    """
    return _DATASET_EXTENSION.get(sniff_format(data), "bin")
