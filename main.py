import argparse
from pathlib import Path

from image_module import display_frames


SHOW_IMAGES = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fighter punch capture starter")
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

    if SHOW_IMAGES:
        display_frames("datasets/standing", stop_key=args.stop_key)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
