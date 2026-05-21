from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import experiments.audio_effects 
from experiments.api_client import SpeakKapClient
from experiments.dataset import SUPPORTED_AUDIO, SpeakerSplit, build_splits, write_manifest
from experiments.metrics import compute_metrics


PASSWORD = "password"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run speaker-authentication experiments.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:5000")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--background-root", type=Path)
    parser.add_argument("--rir-root", type=Path)
    parser.add_argument("--speakers", type=int, default=100)
    parser.add_argument("--enroll-files", type=int, default=4)
    parser.add_argument("--baseline-trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-codec", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = build_splits(
        dataset_root=args.dataset_root,
        speaker_count=args.speakers,
        enroll_per_speaker=args.enroll_files,
        seed=args.seed,
    )
    write_manifest(splits, args.output_dir / "manifest.csv")

    client = SpeakKapClient(args.api_url)
    try:
        # zad 1
        enroll_users(client, splits)
        trials = make_balanced_trials(splits, args.baseline_trials, rng)
        run_experiment(client, "01_baseline", trials, args.output_dir)
        # zad 2
        run_amplitude(client, trials[:500], args.output_dir, rng)
        # zad 3
        run_downsampling(client, trials[:200], args.output_dir)
        # zad 4
        run_gaussian_noise(client, trials[:100], args.output_dir, args.seed)
        # zad 5
        if args.background_root:
            run_background_noise(client, trials[:100], args.output_dir, args.background_root, rng)
        # zad 6
        if not args.skip_codec:
            run_codec(client, trials[:100], args.output_dir)
        # zad 7
        if args.rir_root:
            run_reverb(client, trials[:100], args.output_dir, args.rir_root, rng)
    finally:
        client.close()


def enroll_users(client: SpeakKapClient, splits: list[SpeakerSplit]) -> None:
    client.clear()
    for split in splits:
        client.register(split.speaker_id, PASSWORD, split.enroll_files)


def make_balanced_trials(
    splits: list[SpeakerSplit],
    count: int,
    rng: random.Random,
) -> list[dict]:
    positives = []
    negatives = []
    split_by_id = {split.speaker_id: split for split in splits}
    speaker_ids = list(split_by_id)

    for split in splits:
        for file_path in split.test_files:
            positives.append(
                {
                    "claimed_speaker": split.speaker_id,
                    "true_speaker": split.speaker_id,
                    "file": file_path,
                    "expected_accept": True,
                }
            )
            impostor = rng.choice([sid for sid in speaker_ids if sid != split.speaker_id])
            negatives.append(
                {
                    "claimed_speaker": impostor,
                    "true_speaker": split.speaker_id,
                    "file": file_path,
                    "expected_accept": False,
                }
            )

    half = count // 2
    return _sample(positives, half, rng) + _sample(negatives, count - half, rng)


def run_amplitude(
    client: SpeakKapClient,
    trials: list[dict],
    output_dir: Path,
    rng: random.Random,
) -> None:
    factors = [25.0, 1.0, 0.04]
    transformed = []
    work_dir = output_dir / "audio" / "02_amplitude"
    for index, trial in enumerate(trials):
        factor = rng.choice(factors)
        dst = work_dir / f"{index:04d}_amp_{factor:g}.wav"
        transformed.append({**trial, "file": audio_effects.amplitude(trial["file"], dst, factor)})
    run_experiment(client, "02_amplitude", transformed, output_dir)


def run_downsampling(client: SpeakKapClient, trials: list[dict], output_dir: Path) -> None:
    for mode, transform in (
        ("naive", audio_effects.naive_subsample),
        ("interpolated", audio_effects.interpolated_downsample),
    ):
        for factor in (2, 5, 10):
            transformed = []
            work_dir = output_dir / "audio" / f"03_downsample_{mode}_{factor}x"
            for index, trial in enumerate(trials):
                dst = work_dir / f"{index:04d}.wav"
                transformed.append({**trial, "file": transform(trial["file"], dst, factor)})
            run_experiment(client, f"03_downsample_{mode}_{factor}x", transformed, output_dir)


def run_gaussian_noise(
    client: SpeakKapClient,
    trials: list[dict],
    output_dir: Path,
    seed: int,
) -> None:
    for snr_db in (40, 20, 10):
        transformed = []
        work_dir = output_dir / "audio" / f"04_gaussian_{snr_db}db"
        for index, trial in enumerate(trials):
            dst = work_dir / f"{index:04d}.wav"
            transformed.append(
                {
                    **trial,
                    "file": audio_effects.gaussian_noise(
                        trial["file"], dst, snr_db, seed + index
                    ),
                }
            )
        run_experiment(client, f"04_gaussian_{snr_db}db", transformed, output_dir)


def run_background_noise(
    client: SpeakKapClient,
    trials: list[dict],
    output_dir: Path,
    background_root: Path,
    rng: random.Random,
) -> None:
    noises = [p for p in background_root.rglob("*") if p.suffix.lower() in SUPPORTED_AUDIO]
    if not noises:
        raise ValueError(f"No background audio files found in {background_root}.")
    for snr_db in (20, 10, 0):
        transformed = []
        work_dir = output_dir / "audio" / f"05_background_{snr_db}db"
        for index, trial in enumerate(trials):
            dst = work_dir / f"{index:04d}.wav"
            transformed.append(
                {
                    **trial,
                    "file": audio_effects.background_noise(
                        trial["file"], rng.choice(noises), dst, snr_db
                    ),
                }
            )
        run_experiment(client, f"05_background_{snr_db}db", transformed, output_dir)


def run_codec(client: SpeakKapClient, trials: list[dict], output_dir: Path) -> None:
    settings = [
        ("libmp3lame", "32k"),
        ("libmp3lame", "96k"),
        ("libmp3lame", "192k"),
        ("aac", "32k"),
        ("aac", "96k"),
        ("aac", "192k"),
        ("libopus", "24k"),
        ("libopus", "64k"),
        ("libopus", "128k"),
    ]
    for codec, bitrate in settings:
        transformed = []
        name = f"06_codec_{codec}_{bitrate}"
        work_dir = output_dir / "audio" / name
        for index, trial in enumerate(trials):
            dst = work_dir / f"{index:04d}.wav"
            transformed.append(
                {
                    **trial,
                    "file": audio_effects.codec_roundtrip(trial["file"], dst, codec, bitrate),
                }
            )
        run_experiment(client, name, transformed, output_dir)


def run_reverb(
    client: SpeakKapClient,
    trials: list[dict],
    output_dir: Path,
    rir_root: Path,
    rng: random.Random,
) -> None:
    rirs = [p for p in rir_root.rglob("*") if p.suffix.lower() in {".wav", ".flac"}]
    if not rirs:
        raise ValueError(f"No RIR audio files found in {rir_root}.")
    transformed = []
    work_dir = output_dir / "audio" / "07_reverb"
    for index, trial in enumerate(trials):
        dst = work_dir / f"{index:04d}.wav"
        transformed.append(
            {**trial, "file": audio_effects.reverb(trial["file"], rng.choice(rirs), dst)}
        )
    run_experiment(client, "07_reverb", transformed, output_dir)


def run_experiment(
    client: SpeakKapClient,
    name: str,
    trials: list[dict],
    output_dir: Path,
) -> None:
    rows = []
    for index, trial in enumerate(trials):
        result = client.authenticate(trial["claimed_speaker"], PASSWORD, trial["file"])
        rows.append(
            {
                "trial": index,
                "experiment": name,
                "claimed_speaker": trial["claimed_speaker"],
                "true_speaker": trial["true_speaker"],
                "file": str(trial["file"]),
                "expected_accept": trial["expected_accept"],
                **{k: v for k, v in result.items() if k != "payload"},
            }
        )

    metrics = compute_metrics(rows)
    write_rows(output_dir / f"{name}.csv", rows)
    (output_dir / f"{name}_metrics.json").write_text(
        json.dumps(asdict(metrics), indent=2),
        encoding="utf-8",
    )


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sample(items: list[dict], count: int, rng: random.Random) -> list[dict]:
    if len(items) >= count:
        return rng.sample(items, count)
    return [rng.choice(items) for _ in range(count)]


if __name__ == "__main__":
    main()
