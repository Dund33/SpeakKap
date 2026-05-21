from __future__ import annotations

import csv
import random
import torchaudio
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_AUDIO = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


@dataclass(frozen=True)
class SpeakerSplit:
    speaker_id: str
    enroll_files: list[Path]
    test_files: list[Path]


def discover_speakers(dataset_root: Path) -> dict[str, list[Path]]:
    speakers: dict[str, list[Path]] = {}
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO:
            continue
        speaker_id = path.relative_to(dataset_root).parts[0]
        speakers.setdefault(speaker_id, []).append(path)

    return {
        speaker_id: sorted(files)
        for speaker_id, files in sorted(speakers.items())
        if len(files) >= 2
    }


def build_splits(
    dataset_root: Path,
    speaker_count: int,
    enroll_per_speaker: int,
    seed: int,
) -> list[SpeakerSplit]:
    rng = random.Random(seed)
    speakers = discover_speakers(dataset_root)
    eligible = [
        (speaker_id, files)
        for speaker_id, files in speakers.items()
        if len(files) > enroll_per_speaker
    ]
    if len(eligible) < speaker_count:
        raise ValueError(
            f"Need {speaker_count} speakers with at least "
            f"{enroll_per_speaker + 1} files each, found {len(eligible)}."
        )

    rng.shuffle(eligible)
    splits = []
    for speaker_id, files in sorted(eligible[:speaker_count]):
        shuffled = list(files)
        rng.shuffle(shuffled)
        splits.append(
            SpeakerSplit(
                speaker_id=speaker_id,
                enroll_files=sorted(shuffled[:enroll_per_speaker]),
                test_files=sorted(shuffled[enroll_per_speaker:]),
            )
        )
    return splits


def write_manifest(splits: list[SpeakerSplit], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["speaker_id", "partition", "path", "duration_seconds", "sample_rate"],
        )
        writer.writeheader()
        for split in splits:
            for partition, files in (
                ("enroll", split.enroll_files),
                ("test", split.test_files),
            ):
                for path in files:
                    metadata = torchaudio.info(str(path))
                    duration = metadata.num_frames / metadata.sample_rate
                    writer.writerow(
                        {
                            "speaker_id": split.speaker_id,
                            "partition": partition,
                            "path": str(path),
                            "duration_seconds": f"{duration:.3f}",
                            "sample_rate": metadata.sample_rate,
                        }
                    )
