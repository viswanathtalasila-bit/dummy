from pathlib import Path

import cv2
from rtmlib import Wholebody


DATASET_ROOT = Path(__file__).resolve().parent / "datasets"


def find_image_subfolder(root: Path) -> Path:
    for top_level_dir in sorted(root.iterdir()):
        if not top_level_dir.is_dir():
            continue
        for subfolder in sorted(top_level_dir.iterdir()):
            if not subfolder.is_dir():
                continue
            image_files = sorted([path for path in subfolder.glob("*.png") if path.is_file()])
            if image_files:
                return subfolder
    raise FileNotFoundError(f"No image subfolder found under {root}")


def get_image_paths(subfolder: Path):
    return sorted([path for path in subfolder.glob("*.png") if path.is_file()])


def main(image_path: str | None = None) -> int:
    # -----------------------------
    # Initialize RTMPose
    # -----------------------------
    pose = Wholebody(
        to_openpose=False,
        mode="balanced",  # options: performance, balanced, accuracy
    )

    if image_path is None:
        selected_subfolder = find_image_subfolder(DATASET_ROOT)
        image_paths = get_image_paths(selected_subfolder)
        print(f"Using image folder: {selected_subfolder}")
    else:
        image_path_obj = Path(image_path)
        if image_path_obj.is_dir():
            image_paths = get_image_paths(image_path_obj)
        else:
            image_paths = [image_path_obj]

    if not image_paths:
        raise FileNotFoundError(f"No images found for {image_path or 'selected dataset folder'}")

    for image_file in image_paths:
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Skipping unreadable image: {image_file}")
            continue

        # -----------------------------
        # Run pose estimation
        # -----------------------------
        keypoints, scores = pose(image)
        print(f"Detected {len(keypoints)} person(s) in {image_file.name}")

        # -----------------------------
        # Draw detected keypoints
        # -----------------------------
        for person, person_scores in zip(keypoints, scores):
            for (x, y), conf in zip(person, person_scores):
                if conf > 0.3:
                    cv2.circle(
                        image,
                        (int(x), int(y)),
                        4,
                        (0, 255, 0),
                        -1,
                    )

        # -----------------------------
        # Show result
        # -----------------------------
        cv2.imshow("RTMPose", image)
        if cv2.waitKey(500) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())