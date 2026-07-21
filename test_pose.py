import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from rtmlib import Wholebody
from rtmlib.visualization.skeleton import (
    animal17,
    coco17,
    coco25,
    coco133,
    halpe26,
    hand21,
    openpose18,
    openpose134,
)


DATASET_ROOT = Path(__file__).resolve().parent / "datasets"
OUTPUT_CSV = DATASET_ROOT / "punching_1" / "pose_keypoints.csv"


SKELETON_BY_KEYPOINT_COUNT_MMPPOSE: Dict[int, dict] = {
    17: coco17,
    21: hand21,
    25: coco25,
    26: halpe26,
    133: coco133,
}

SKELETON_BY_KEYPOINT_COUNT_OPENPOSE: Dict[int, dict] = {
    17: animal17,
    18: openpose18,
    26: halpe26,
    134: openpose134,
}


def get_joint_names(num_keypoints: int, to_openpose: bool) -> List[str]:
    skeleton_lookup = (
        SKELETON_BY_KEYPOINT_COUNT_OPENPOSE
        if to_openpose
        else SKELETON_BY_KEYPOINT_COUNT_MMPPOSE
    )
    skeleton = skeleton_lookup.get(num_keypoints)
    if skeleton is None:
        return [f"joint_{index}" for index in range(num_keypoints)]

    keypoint_info = skeleton["keypoint_info"]
    return [keypoint_info[index]["name"] for index in range(num_keypoints)]


def derive_pose_label(subfolder_name: str) -> str:
    normalized = subfolder_name.lower().replace("-", "_").replace(" ", "_")
    if "jab" in normalized:
        return "jab"
    if "cross" in normalized:
        return "right cross"
    if "hook" in normalized:
        if "left" in normalized:
            return "left hook"
        if "right" in normalized:
            return "right hook"
    if "left" in normalized:
        return "left"
    if "right" in normalized:
        return "right"
    return normalized.replace("_", " ")


def collect_subfolders(root: Path) -> List[Path]:
    subfolders = []
    for top_level_dir in sorted(root.iterdir()):
        if not top_level_dir.is_dir():
            continue
        for subfolder in sorted(top_level_dir.iterdir()):
            if subfolder.is_dir() and any(path.is_file() for path in subfolder.glob("*.png")):
                subfolders.append(subfolder)
    return subfolders


def get_image_paths(subfolder: Path):
    return sorted([path for path in subfolder.glob("*.png") if path.is_file()])


def build_frame_row(keypoints, scores, frame_number: int, pose_label: str,
                    joint_names_for_model: List[str]) -> List[object]:
    joint_names = []
    xs = []
    ys = []
    confidences = []

    for person_index, (person, person_scores) in enumerate(zip(keypoints, scores)):
        for joint_index, ((x, y), conf) in enumerate(zip(person, person_scores)):
            if joint_index < len(joint_names_for_model):
                joint_name = joint_names_for_model[joint_index]
            else:
                joint_name = f"joint_{joint_index}"
            joint_names.append(f"person_{person_index}_{joint_name}")
            xs.append(float(x))
            ys.append(float(y))
            confidences.append(float(conf))

    return [
        frame_number,
        pose_label,
        json.dumps(joint_names),
        json.dumps(xs),
        json.dumps(ys),
        json.dumps(confidences),
    ]


def write_pose_csv(rows: List[List[object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame number", "pose", "joint name", "x", "y", "confidence"])
        writer.writerows(rows)


def main(
    image_path: Optional[str] = None,
    show_images: bool = False,
    write_csv: bool = True,
) -> int:
    # -----------------------------
    # Initialize RTMPose
    # -----------------------------
    pose = Wholebody(
        to_openpose=False,
        mode="balanced",  # options: performance, balanced, accuracy
    )
    to_openpose = False

    if image_path is None:
        subfolders = collect_subfolders(DATASET_ROOT)
        if not subfolders:
            raise FileNotFoundError(f"No image subfolders found under {DATASET_ROOT}")
    else:
        image_path_obj = Path(image_path)
        if image_path_obj.is_dir():
            subfolders = [image_path_obj]
        else:
            subfolders = [image_path_obj.parent]

    all_rows = []
    skipped_images = 0
    joint_names_cache: Dict[int, List[str]] = {}
    for subfolder in subfolders:
        if not subfolder.exists():
            continue

        pose_label = derive_pose_label(subfolder.name)
        image_paths = get_image_paths(subfolder) if subfolder.is_dir() else [subfolder]
        if not image_paths:
            continue

        print(f"Processing folder: {subfolder}")
        for image_file in image_paths:
            try:
                image = cv2.imread(str(image_file))
                if image is None:
                    skipped_images += 1
                    print(f"Skipping unreadable image: {image_file}")
                    continue

                frame_number = int(image_file.stem.split("_")[-1]) if image_file.stem.split("_")[-1].isdigit() else 0
                keypoints, scores = pose(image)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                skipped_images += 1
                print(f"Skipping {image_file.name}: {exc}")
                continue

            print(f"Detected {len(keypoints)} person(s) in {image_file.name}")
            if hasattr(keypoints, "shape") and len(keypoints.shape) >= 2:
                num_keypoints = int(keypoints.shape[1])
                if num_keypoints not in joint_names_cache:
                    joint_names_cache[num_keypoints] = get_joint_names(
                        num_keypoints=num_keypoints,
                        to_openpose=to_openpose,
                    )
            else:
                num_keypoints = 0

            row = build_frame_row(
                keypoints,
                scores,
                frame_number,
                pose_label,
                joint_names_cache.get(num_keypoints, []),
            )
            all_rows.append(row)

            if show_images:
                # Draw detected points for preview
                for person, person_scores in zip(keypoints, scores):
                    for (x, y), conf in zip(person, person_scores):
                        if conf > 0.3:
                            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)

                cv2.imshow("RTMPose", image)
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break

        if write_csv:
            write_pose_csv(all_rows, OUTPUT_CSV)
            print(f"Updated CSV after folder {subfolder.name}: {len(all_rows)} total pose rows")

    if show_images:
        cv2.destroyAllWindows()

    if write_csv:
        print(f"Wrote {len(all_rows)} pose rows to {OUTPUT_CSV} (skipped {skipped_images} images)")
    else:
        print(f"Processed {len(all_rows)} pose rows without writing CSV (skipped {skipped_images} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())