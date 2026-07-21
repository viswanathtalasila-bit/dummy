import argparse
from pathlib import Path

from image_module import display_frames
from test_pose import main as run_pose_estimation
from video_frame_extractor import extract_frames_for_intervals


def prompt_pose_estimation_csv_output() -> bool:
    print("Choose pose estimation output:")
    print("1) Show images with detected joints only")
    print("2) Write CSV output only")

    while True:
        selection = input("Enter 1 or 2: ").strip()
        if selection == "1":
            return False
        if selection == "2":
            return True
        print("Invalid selection. Enter 1 or 2.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fighter punch capture starter")
    parser.add_argument(
        "--mode",
        choices=["display", "extract", "pose_estimation"],
        default="pose_estimation",
        help="Choose whether to display frames, extract timed frames from a video, or run pose estimation",
    )
    parser.add_argument("--source", default="0", help="Video source index or path")
    parser.add_argument("--output", default="output", help="Output folder for captured frames")
    parser.add_argument("--stop-key", default="q", help="Keyboard key that stops frame display")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Capture source: {args.source}")
    print(f"Output directory: {output_dir.resolve()}")

    if args.mode == "extract":
        extract_frames_for_intervals(video_path=args.source, output_dir=args.output)
        return 0

    if args.mode == "pose_estimation":
        write_csv = prompt_pose_estimation_csv_output()
        run_pose_estimation(show_images=not write_csv, write_csv=write_csv)
        return 0

    if args.mode == "display":
        display_frames("datasets/punching_1/power_right_hooks/", stop_key=args.stop_key)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
