import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class SummaryTests(unittest.TestCase):
    def test_markdown_table_does_not_require_tabulate(self):
        from summarize_results import markdown_table

        frame = pd.DataFrame({"Method": ["jin"], "Accuracy mean": [0.75]})

        text = markdown_table(frame)

        self.assertIn("| Method | Accuracy mean |", text)
        self.assertIn("| jin | 0.75 |", text)


if __name__ == "__main__":
    unittest.main()
