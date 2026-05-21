from __future__ import annotations

import math
import shutil
import subprocess
import uuid
from pathlib import Path


def load_audio(path: Path):
    torch = _torch()
    torchaudio = _torchaudio()
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.to(torch.float32), sample_rate


def save_audio(path: Path, waveform, sample_rate: int) -> Path:
    torchaudio = _torchaudio()
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = waveform.clamp(-1.0, 1.0).cpu()
    torchaudio.save(str(path), waveform, sample_rate)
    return path


def copy_audio(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def amplitude(src: Path, dst: Path, factor: float) -> Path:
    waveform, sample_rate = load_audio(src)
    return save_audio(dst, waveform * factor, sample_rate)


def naive_subsample(src: Path, dst: Path, factor: int) -> Path:
    waveform, sample_rate = load_audio(src)
    reduced = waveform[:, ::factor]
    return save_audio(dst, reduced, max(1, sample_rate // factor))


def interpolated_downsample(src: Path, dst: Path, factor: int) -> Path:
    torchaudio = _torchaudio()
    waveform, sample_rate = load_audio(src)
    new_sample_rate = max(1, sample_rate // factor)
    transformed = torchaudio.functional.resample(waveform, sample_rate, new_sample_rate)
    return save_audio(dst, transformed, new_sample_rate)


def gaussian_noise(src: Path, dst: Path, snr_db: float, seed: int) -> Path:
    torch = _torch()
    waveform, sample_rate = load_audio(src)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
    mixed = _mix_at_snr(waveform, noise, snr_db)
    return save_audio(dst, mixed, sample_rate)


def background_noise(src: Path, noise_src: Path, dst: Path, snr_db: float) -> Path:
    torchaudio = _torchaudio()
    waveform, sample_rate = load_audio(src)
    noise, noise_sample_rate = load_audio(noise_src)
    if noise_sample_rate != sample_rate:
        noise = torchaudio.functional.resample(noise, noise_sample_rate, sample_rate)
    noise = _fit_length(noise, waveform.shape[1])
    mixed = _mix_at_snr(waveform, noise, snr_db)
    return save_audio(dst, mixed, sample_rate)


def reverb(src: Path, rir_src: Path, dst: Path) -> Path:
    torchaudio = _torchaudio()
    torch_f = _torch_functional()
    waveform, sample_rate = load_audio(src)
    rir, rir_sample_rate = load_audio(rir_src)
    if rir_sample_rate != sample_rate:
        rir = torchaudio.functional.resample(rir, rir_sample_rate, sample_rate)
    rir = rir / (rir.abs().max().clamp_min(1e-8))
    convolved = torch_f.conv1d(
        waveform.unsqueeze(0),
        rir.flip(dims=[1]).unsqueeze(0),
        padding=rir.shape[1] - 1,
    ).squeeze(0)
    convolved = convolved[:, : waveform.shape[1]]
    return save_audio(dst, convolved, sample_rate)


def codec_roundtrip(src: Path, dst: Path, codec: str, bitrate: str) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.stem}-{uuid.uuid4().hex}.{_codec_extension(codec)}")

    encode_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-c:a",
        codec,
        "-b:a",
        bitrate,
        str(tmp),
    ]
    decode_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(tmp),
        str(dst),
    ]
    try:
        subprocess.run(encode_cmd, check=True)
        subprocess.run(decode_cmd, check=True)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def _fit_length(waveform, length: int):
    if waveform.shape[1] >= length:
        return waveform[:, :length]
    repeats = math.ceil(length / waveform.shape[1])
    return waveform.repeat(1, repeats)[:, :length]


def _mix_at_snr(signal, noise, snr_db: float):
    torch = _torch()
    signal_power = signal.pow(2).mean().clamp_min(1e-12)
    noise_power = noise.pow(2).mean().clamp_min(1e-12)
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    scaled_noise = noise * torch.sqrt(target_noise_power / noise_power)
    return signal + scaled_noise


def _codec_extension(codec: str) -> str:
    if codec == "libmp3lame":
        return "mp3"
    if codec in {"aac", "libfdk_aac"}:
        return "m4a"
    if codec in {"libopus", "opus"}:
        return "opus"
    return "encoded"


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Audio experiments require torch. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return torch


def _torch_functional():
    try:
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError(
            "Reverb experiments require torch. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return functional


def _torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError(
            "Audio experiments require torchaudio. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return torchaudio
