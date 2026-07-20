import shutil
import tempfile
import unittest
from pathlib import Path

from main import build_parser
from video_frame_extractor import DEFAULT_VIDEO_PATH, extract_frames_for_intervals


class ProjectSmokeTest(unittest.TestCase):
    def test_parser_supports_source_and_output(self):
        parser = build_parser()
        args = parser.parse_args(["--source", "0", "--output", "output"])

        self.assertEqual(args.source, "0")
        self.assertEqual(args.output, "output")

    def test_parser_supports_stop_key(self):
        parser = build_parser()
        args = parser.parse_args(["--source", "0", "--output", "output", "--stop-key", "q"])

        self.assertEqual(args.stop_key, "q")

    def test_parser_supports_mode_switch(self):
        parser = build_parser()
        args = parser.parse_args(["--mode", "extract", "--source", "datasets/punching_1/punching_1.mp4"])

        self.assertEqual(args.mode, "extract")
        self.assertEqual(args.source, "datasets/punching_1/punching_1.mp4")

    def test_parser_defaults_to_pose_mode(self):
        parser = build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.mode, "pose")

    def test_extract_uses_default_video_when_source_is_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            created_dirs = extract_frames_for_intervals(
                video_path="0",
                output_dir=output_dir,
                intervals_by_comment={"jab only": (0, 1)},
            )

            self.assertEqual(created_dirs["jab only"], output_dir / "jab_only")
            self.assertTrue((output_dir / "jab_only").exists())
            self.assertTrue(DEFAULT_VIDEO_PATH.exists())


if __name__ == "__main__":
    unittest.main()
