import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "make_catalog", ROOT / "tools" / "lemonade" / "make_catalog.py")
make_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_catalog)


class MakeCatalogTest(unittest.TestCase):
    def entry(self, file_name):
        return {"name": "M", "repo": "o/r", "file": file_name, "mmproj": None,
                "draft": None, "labels": ["chat"], "size_gb": 1.0,
                "superroc_args": ["-t", "0"]}

    def test_single_file_checkpoint(self):
        catalog, args = make_catalog.build([self.entry("M-Q4.gguf")])
        self.assertEqual(catalog["M"]["checkpoints"]["main"], "o/r:M-Q4.gguf")
        self.assertEqual(args, {"M-Q4.gguf": ["-t", "0"]})

    def test_split_file_checkpoint_is_the_series(self):
        catalog, args = make_catalog.build([self.entry("M-Q4-00001-of-00002.gguf")])
        self.assertEqual(catalog["M"]["checkpoints"]["main"], "o/r:M-Q4")
        # The launcher sees the first part's file name.
        self.assertEqual(args, {"M-Q4-00001-of-00002.gguf": ["-t", "0"]})


if __name__ == "__main__":
    unittest.main()
