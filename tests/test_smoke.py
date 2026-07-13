import unittest

from main import build_parser


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


if __name__ == "__main__":
    unittest.main()
