import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class CoreUtilityTests(unittest.TestCase):
    def test_mfcc_with_deltas_returns_39_coefficients(self):
        from sota_baselines.mfcc import mfcc_with_deltas

        sample_rate = 16000
        t = np.arange(sample_rate, dtype=np.float64) / sample_rate
        signal = 0.25 * np.sin(2 * np.pi * 440.0 * t)

        features = mfcc_with_deltas(signal, sample_rate)

        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], 39)
        self.assertGreater(features.shape[0], 20)
        self.assertTrue(np.isfinite(features).all())

    def test_jin_same_value_filter_removes_features_over_threshold(self):
        from sota_baselines.jin_features import same_value_filter

        frame = pd.DataFrame(
            {
                "constant": [1, 1, 1, 1, 2],
                "mostly_constant": [0, 0, 0, 0, 0],
                "varied": [0, 1, 2, 3, 4],
            }
        )

        kept = same_value_filter(frame, threshold=0.70)

        self.assertEqual(kept, ["varied"])

    def test_leave_one_device_out_keeps_test_device_disjoint(self):
        from sota_baselines.evaluation import leave_one_group_splits

        labels = np.array(["a", "a", "b", "b", "c", "c"])
        groups = np.array(["d1", "d1", "d2", "d2", "d3", "d3"])

        splits = list(leave_one_group_splits(labels, groups))

        self.assertEqual(len(splits), 3)
        for _, train_idx, test_idx in splits:
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            self.assertTrue(train_groups.isdisjoint(test_groups))
            self.assertEqual(len(test_groups), 1)


if __name__ == "__main__":
    unittest.main()
