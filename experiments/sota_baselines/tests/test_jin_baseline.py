import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class JinBaselineTests(unittest.TestCase):
    def test_fit_predict_jin_svm_filters_and_predicts(self):
        from sota_baselines.jin_baseline import fit_predict_jin_svm

        train = pd.DataFrame(
            {
                "file": [f"tr{i}.m4a" for i in range(8)],
                "label": ["a", "a", "a", "a", "b", "b", "b", "b"],
                "constant": [1] * 8,
                "signal": [0.0, 0.1, 0.0, 0.2, 1.0, 1.1, 1.0, 1.2],
            }
        )
        test = pd.DataFrame(
            {
                "file": ["te0.m4a", "te1.m4a"],
                "label": ["a", "b"],
                "constant": [1, 1],
                "signal": [0.05, 1.05],
            }
        )

        result = fit_predict_jin_svm(
            train,
            test,
            feature_columns=["constant", "signal"],
            portions=(1.0,),
            inner_cv_splits=0,
        )

        self.assertEqual(result.predictions.tolist(), ["a", "b"])
        self.assertEqual(result.selected_columns, ["signal"])
        self.assertEqual(result.selected_portion, 1.0)

    def test_fit_predict_jin_tuned_svm_reports_selected_hyperparameters(self):
        from sota_baselines.jin_baseline import fit_predict_jin_tuned_svm

        train = pd.DataFrame(
            {
                "file": [f"tr{i}.m4a" for i in range(12)],
                "label": ["a"] * 6 + ["b"] * 6,
                "constant": [1] * 12,
                "signal": [0.0, 0.1, 0.2, 0.05, 0.15, 0.25, 1.0, 1.1, 1.2, 1.05, 1.15, 1.25],
                "weak": [0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3],
            }
        )
        test = pd.DataFrame(
            {
                "file": ["te0.m4a", "te1.m4a"],
                "label": ["a", "b"],
                "constant": [1, 1],
                "signal": [0.12, 1.12],
                "weak": [0.2, 0.4],
            }
        )

        result = fit_predict_jin_tuned_svm(
            train,
            test,
            feature_columns=["constant", "signal", "weak"],
            portions=(0.5, 1.0),
            c_values=(0.1, 1.0),
            gamma_values=("scale", "auto"),
            inner_cv_splits=3,
            random_state=7,
        )

        self.assertEqual(result.predictions.tolist(), ["a", "b"])
        self.assertIn(result.selected_c, (0.1, 1.0))
        self.assertIn(result.selected_gamma, ("scale", "auto"))
        self.assertIn(result.selected_portion, (0.5, 1.0))
        self.assertGreaterEqual(result.validation_score, 0.0)


if __name__ == "__main__":
    unittest.main()
