from __future__ import annotations

import re
from pathlib import Path

import cv2

DEFAULT_VIDEO_PATH = Path("datasets/punching_1/punching_1.mp4")
DEFAULT_OUTPUT_DIR = Path("datasets/punching_1")


def resolve_video_path(video_path: str | Path = DEFAULT_VIDEO_PATH) -> Path:
    """Resolve a CLI source into an existing video file path.

    The CLI uses "0" as the default source value. For frame extraction, a
    numeric source should fall back to the bundled dataset video when present.
    """
    candidate = Path(video_path)
    if candidate.exists():
        return candidate

    if str(video_path).strip().isdigit() and DEFAULT_VIDEO_PATH.exists():
        return DEFAULT_VIDEO_PATH

    return candidate


INTERVALS_BY_COMMENT: dict[str, tuple[int, int]] = {
    "1-2, 1-2 non stop": (59, 120),
    "jab only": (144, 202),
    "cross only": (220, 284),
    "1-2, 1-2, non stop": (300, 360),
    "power left hooks": (400, 440),
    "power right hooks": (460, 523),
}


def sanitize_folder_name(comment_name: str) -> str:
    """Convert a comment label into a usable directory name."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", comment_name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.lower()


def extract_frames_for_intervals(
    video_path: str | Path = DEFAULT_VIDEO_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    intervals_by_comment: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Path]:
    """Extract all frames for each named interval and store them into separate folders."""
    video_file = resolve_video_path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    intervals = intervals_by_comment or INTERVALS_BY_COMMENT
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_file}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    created_dirs: dict[str, Path] = {}
    frame_summaries: list[tuple[str, int, int]] = []

    for comment_name, (start_seconds, end_seconds) in intervals.items():
        folder_name = sanitize_folder_name(comment_name)
        folder_path = output_root / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        created_dirs[comment_name] = folder_path

        start_frame = max(0, int(round(start_seconds * fps)))
        end_frame = min(total_frames, int(round(end_seconds * fps)))

        if start_frame >= end_frame:
            continue

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        saved_count = 0

        while current_frame < end_frame:
            ok, frame = capture.read()
            if not ok:
                break

            output_path = folder_path / f"frame_{current_frame:06d}.png"
            cv2.imwrite(str(output_path), frame)
            saved_count += 1
            current_frame += 1

        frame_summaries.append((comment_name, saved_count, current_frame - start_frame))

    capture.release()
    return created_dirs


def main() -> int:
    created_dirs = extract_frames_for_intervals()
    for comment_name, directory in created_dirs.items():
        print(f"Saved frames for '{comment_name}' in {directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
