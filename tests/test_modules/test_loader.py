import os
import shutil
import unittest
from tempfile import mkdtemp

import pandas as pd

from corebehrt.modules.features.loader import FormattedDataLoader
from corebehrt.constants.data import (
    PID_COL,
    CONCEPT_COL,
)


class TestFormattedDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = mkdtemp()

        # Create test concept data
        concept_data = pd.DataFrame(
            {
                PID_COL: [1, 2],
                "time": pd.to_datetime(["2020-01-01", "2010-01-01"]),
                CONCEPT_COL: ["A", "B"],
                "numeric_value": ["1", "2"],
            }
        )
        concept_path = os.path.join(self.test_dir, "1.parquet")
        concept_data.to_parquet(concept_path)

    def tearDown(self):
        # Remove the temporary directory and files
        shutil.rmtree(self.test_dir)

    def test_load(self):
        # Initialize the FormattedDataLoader
        concept_path = os.path.join(self.test_dir, "1.parquet")
        loader = FormattedDataLoader(path=concept_path)

        # Load the data
        concepts = loader.load()

        # Verify the concepts data
        expected_concepts_data = pd.DataFrame(
            {
                PID_COL: [1, 2],
                "time": pd.to_datetime(["2020-01-01", "2010-01-01"]),
                CONCEPT_COL: pd.Series(["A", "B"]),
                "numeric_value": ["1", "2"],
            }
        )
        pd.testing.assert_frame_equal(concepts, expected_concepts_data)


if __name__ == "__main__":
    unittest.main()
