"""Format dashboard training artifacts for posting.

Renames video/pdf/audio to {nn}_{use_case}.* and clips the Gemini Notebook
end card from each video. Ignores .pptx.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

SRC = Path(r"I:\My Drive\PGx_Dashboard_Use_Cases\dashboard_training")
OUT = SRC / "posting"

CASES = [
    ("01", "cohort_risk", "uc1"),
    ("02", "scenario_analysis", "uc2"),
    ("03", "density_bin_exploration", "uc3"),
    ("04", "feature_importance", "uc4"),
    ("05", "pattern_process", "uc5"),
    ("06", "claims_pgx_card", "uc6"),
    ("07", "personalized_pgx_card", "uc7"),
    ("08", "cohort_vs_card", "uc8"),
    ("09", "documentation", "uc9"),
]


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    )
    return float(out.strip())


def mean_luma(img_path: Path) -> float:
    im = Image.open(img_path).convert("L")
    hist = im.histogram()
    total = sum(hist)
    if not total:
        return 0.0
    return sum(i * c for i, c in enumerate(hist)) / total


def detect_endcard_start(video: Path, scratch: Path) -> float:
    """Return timestamp (seconds) to cut before the Gemini Notebook end card."""
    duration = ffprobe_duration(video)
    window = min(16.0, max(6.0, duration * 0.05))
    start = max(0.0, duration - window)
    frames_dir = scratch / video.stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(video),
            "-vf",
            "fps=2",
            "-q:v",
            "5",
            str(frames_dir / "%03d.jpg"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return max(0.0, duration - 4.0)

    # White fade / empty card before the Gemini Notebook logo.
    cut_offset = None
    for i, frame in enumerate(frames):
        if mean_luma(frame) >= 245:
            cut_offset = i / 2.0
            break
    if cut_offset is None:
        # Fallback: drop a typical 4s end card.
        return max(0.0, duration - 4.0)
    # Leave a tiny pad so the last content slide is not clipped mid-fade-in.
    return max(0.0, start + cut_offset - 0.15)


def copy_file(src: Path, dest: Path) -> None:
    dest.write_bytes(src.read_bytes())


def clip_video(src: Path, dest: Path, end_sec: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp4")
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-t",
            f"{end_sec:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(tmp),
        ]
    )
    tmp.replace(dest)


def first_match(folder: Path, suffixes: tuple[str, ...]) -> Path | None:
    hits = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    hits.sort(key=lambda p: p.stat().st_size, reverse=True)
    return hits[0] if hits else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    with tempfile.TemporaryDirectory(prefix="pgx_train_end_") as tmp:
        scratch = Path(tmp)
        for num, slug, folder_name in CASES:
            folder = SRC / folder_name
            stem = f"{num}_{slug}"
            video = first_match(folder, (".mp4",))
            pdf = first_match(folder, (".pdf",))
            audio = first_match(folder, (".m4a", ".mp3", ".wav"))
            row = {"uc": stem, "folder": folder_name}
            if pdf:
                dest = OUT / f"{stem}.pdf"
                copy_file(pdf, dest)
                row["pdf"] = dest.name
            if audio:
                dest = OUT / f"{stem}{audio.suffix.lower()}"
                copy_file(audio, dest)
                row["audio"] = dest.name
            if video:
                cut = detect_endcard_start(video, scratch)
                orig_dur = ffprobe_duration(video)
                dest = OUT / f"{stem}.mp4"
                clip_video(video, dest, cut)
                new_dur = ffprobe_duration(dest)
                row["video"] = dest.name
                row["cut_at_sec"] = round(cut, 3)
                row["orig_sec"] = round(orig_dur, 3)
                row["new_sec"] = round(new_dur, 3)
                row["dropped_sec"] = round(orig_dur - new_dur, 3)
            report.append(row)
            print(json.dumps(row))
    (OUT / "posting_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
