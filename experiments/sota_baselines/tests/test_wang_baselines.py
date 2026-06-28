import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class WangBaselineTests(unittest.TestCase):
    def test_map_adapted_gsv_has_component_times_feature_dimension(self):
        from sklearn.mixture import GaussianMixture
        from sota_baselines.wang_baselines import map_adapted_gsv

        rng = np.random.default_rng(7)
        frames = np.vstack(
            [
                rng.normal(0.0, 0.1, size=(20, 3)),
                rng.normal(3.0, 0.1, size=(20, 3)),
            ]
        )
        ubm = GaussianMixture(n_components=2, covariance_type="diag", random_state=7)
        ubm.fit(frames)

        gsv = map_adapted_gsv(ubm, frames, relevance_factor=8.0)

        self.assertEqual(gsv.shape, (6,))
        self.assertTrue(np.isfinite(gsv).all())

    def test_gmm_ubm_predict_scores_one_label_per_sample(self):
        from sota_baselines.wang_baselines import fit_predict_gmm_ubm

        rng = np.random.default_rng(11)
        train_features = [
            rng.normal(0.0, 0.1, size=(15, 4)),
            rng.normal(0.1, 0.1, size=(15, 4)),
            rng.normal(3.0, 0.1, size=(15, 4)),
            rng.normal(3.1, 0.1, size=(15, 4)),
        ]
        train_labels = np.array(["a", "a", "b", "b"])
        test_features = [
            rng.normal(0.0, 0.1, size=(12, 4)),
            rng.normal(3.0, 0.1, size=(12, 4)),
        ]

        predictions = fit_predict_gmm_ubm(
            train_features,
            train_labels,
            test_features,
            n_components=2,
            random_state=11,
        )

        self.assertEqual(predictions.tolist(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
