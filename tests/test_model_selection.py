import os
import unittest

from cli.classify_audio_files import resolve_model_filename


class ModelSelectionTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("ASI_MODEL_NAME", None)
        os.environ.pop("ASI_MODEL_FILE", None)

    def test_default_model_is_lr(self):
        self.assertEqual(resolve_model_filename(), "trained_lr.pkl")

    def test_voting_model_can_be_selected(self):
        self.assertEqual(resolve_model_filename("voting"), "trained_voting.pkl")

    def test_environment_can_select_model_name(self):
        os.environ["ASI_MODEL_NAME"] = "voting"

        self.assertEqual(resolve_model_filename(), "trained_voting.pkl")


if __name__ == "__main__":
    unittest.main()
