import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DatasetTests(unittest.TestCase):
    def test_merge_feature_tables_preserves_one_row_per_file(self):
        from sota_baselines.dataset import merge_feature_tables

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first = tmp_path / "a.csv"
            second = tmp_path / "b.csv"
            first.write_text("file,x\nf1.m4a,1\nf2.m4a,2\n", encoding="utf-8")
            second.write_text("file,y\nf1.m4a,3\nf2.m4a,4\n", encoding="utf-8")

            merged = merge_feature_tables([first, second])

        self.assertEqual(list(merged["file"]), ["f1.m4a", "f2.m4a"])
        self.assertEqual(list(merged.columns), ["file", "a::x", "b::y"])

    def test_resolve_audio_paths_indexes_nested_files_by_name(self):
        from sota_baselines.dataset import resolve_audio_paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "device"
            nested.mkdir()
            target = nested / "f1.m4a"
            target.write_bytes(b"data")

            resolved = resolve_audio_paths(root, ["f1.m4a"])

        self.assertEqual(resolved["f1.m4a"], target)

    def test_attach_metadata_adds_labels_and_groups(self):
        from sota_baselines.dataset import attach_metadata

        features = pd.DataFrame({"file": ["f1.m4a"], "x": [1.0]})
        metadata = pd.DataFrame(
            {
                "file": ["f1.m4a"],
                "label": ["and_band"],
                "device": ["and1"],
            }
        )

        merged = attach_metadata(features, metadata)

        self.assertEqual(merged.loc[0, "label"], "and_band")
        self.assertEqual(merged.loc[0, "device"], "and1")


if __name__ == "__main__":
    unittest.main()
