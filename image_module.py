from pathlib import Path
from typing import Iterator

import cv2



DATASET_DIR = Path("datasets/walking")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_frames(dataset_dir: str | Path = DATASET_DIR) -> Iterator[cv2.typing.MatLike]:
    """Yield image frames from a local dataset directory."""
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    for image_file in sorted(dataset_path.iterdir(), key=lambda path: path.name):
        if image_file.is_file() and image_file.suffix.lower() in IMAGE_EXTENSIONS:
            frame = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame


def display_frames(dataset_dir: str | Path = DATASET_DIR, delay_seconds: float = 0.5, stop_key: str = "q") -> None:
    """Display frames until the configured stop key is pressed."""
    stop_key_code = ord(stop_key.lower())
    cv2.namedWindow("Dataset Frame", cv2.WINDOW_NORMAL)

    try:
        for frame in load_frames(dataset_dir):
            cv2.imshow("Dataset Frame", frame)
            if cv2.waitKey(max(1, int(delay_seconds * 1000))) & 0xFF == stop_key_code:
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    for frame_index, frame in load_frames():
        print(f"Loaded frame {frame_index}: shape={frame.shape}")
