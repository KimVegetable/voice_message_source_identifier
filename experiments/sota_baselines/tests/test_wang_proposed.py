import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for Wang proposed tests")
class WangProposedTests(unittest.TestCase):
    def test_sinc_conv_returns_expected_shape(self):
        import torch
        from sota_baselines.wang_proposed import SincConv1d

        layer = SincConv1d(out_channels=4, kernel_size=31, sample_rate=16000)
        batch = torch.zeros((3, 1, 256), dtype=torch.float32)

        output = layer(batch)

        self.assertEqual(tuple(output.shape), (3, 4, 226))

    def test_make_fixed_segments_pads_and_limits_segments(self):
        from sota_baselines.wang_proposed import make_fixed_segments, normalize_waveform

        waveform = np.arange(10, dtype=np.float32)

        segments = make_fixed_segments(waveform, segment_samples=6, max_segments=3)
        normalized = normalize_waveform(waveform)

        self.assertEqual(segments.shape, (2, 6))
        self.assertTrue(np.allclose(segments[0], normalized[:6]))
        self.assertTrue(np.allclose(segments[-1], normalized[-6:]))

    def test_aggregate_segment_probabilities_uses_mean_probability(self):
        from sota_baselines.wang_proposed import aggregate_segment_probabilities

        probabilities = np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]], dtype=np.float32)

        predicted = aggregate_segment_probabilities(probabilities, classes=np.array(["a", "b"]), mode="mean")

        self.assertEqual(predicted, "b")

    def test_fit_predict_wang_proposed_smoke(self):
        from sota_baselines.wang_proposed import WangProposedConfig, fit_predict_wang_proposed

        sample_rate = 800
        t = np.arange(400, dtype=np.float32) / sample_rate
        class_a = [np.sin(2 * np.pi * 50 * t).astype(np.float32) for _ in range(4)]
        class_b = [np.sin(2 * np.pi * 140 * t).astype(np.float32) for _ in range(4)]
        train_waveforms = class_a[:3] + class_b[:3]
        train_labels = np.array(["a", "a", "a", "b", "b", "b"])
        test_waveforms = [class_a[3], class_b[3]]

        config = WangProposedConfig(
            sample_rate=sample_rate,
            segment_samples=200,
            max_segments_per_file=2,
            n_sinc_filters=4,
            sinc_kernel_size=31,
            conv_channels=4,
            hidden_dim=8,
            dropout=0.0,
            learning_rate=0.01,
            batch_size=4,
            tune_epochs=1,
            final_epochs=2,
        )
        result = fit_predict_wang_proposed(
            train_waveforms,
            train_labels,
            test_waveforms,
            configs=(config,),
            random_state=3,
        )

        self.assertEqual(result.predictions.shape, (2,))
        self.assertTrue(set(result.predictions).issubset({"a", "b"}))
        self.assertEqual(result.selected_config.segment_samples, 200)

    def test_paper_profile_uses_disclosed_wang_architecture_values(self):
        from sota_baselines.wang_proposed import wang_config_grid

        (config,) = wang_config_grid("paper", sample_rate=32000)

        self.assertEqual(config.n_sinc_filters, 80)
        self.assertEqual(config.sinc_kernel_size, 251)
        self.assertEqual(config.sinc_stride, 16)
        self.assertEqual(config.conv_channels, 60)
        self.assertEqual(config.second_conv_channels, 60)
        self.assertEqual(config.first_conv_kernel_size, 5)
        self.assertEqual(config.second_conv_kernel_size, 5)
        self.assertEqual(config.hidden_dim, 2048)
        self.assertEqual(config.dnn_layers, 3)
        self.assertEqual(config.max_segments_per_file, 6)
        self.assertEqual(config.aggregation, "vote")
        self.assertEqual(config.optimizer, "sgd")


if __name__ == "__main__":
    unittest.main()
