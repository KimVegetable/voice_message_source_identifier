import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for MFCC CNN tests")
class TorchBaselineTests(unittest.TestCase):
    def test_sequence_to_image_pads_to_fixed_shape(self):
        from sota_baselines.torch_baselines import sequence_to_image

        sequence = np.ones((5, 39), dtype=np.float32)
        image = sequence_to_image(sequence, max_frames=8)

        self.assertEqual(image.shape, (1, 8, 39))
        self.assertTrue(np.allclose(image[:, :5, :], 1.0))
        self.assertTrue(np.allclose(image[:, 5:, :], 0.0))

    def test_mfcc_cnn_forward_returns_class_logits(self):
        import torch
        from sota_baselines.torch_baselines import MfccCnn

        model = MfccCnn(n_classes=4)
        batch = torch.zeros((3, 1, 32, 39), dtype=torch.float32)

        logits = model(batch)

        self.assertEqual(tuple(logits.shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
