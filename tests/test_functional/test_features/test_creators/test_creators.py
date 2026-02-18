import unittest
import pandas as pd
from datetime import datetime
from pandas import NaT

from corebehrt.functional.features.creators import (
    create_age_in_years,
    create_abspos,
    create_segments,
    create_background,
    sort_features,
)


class TestCreators(unittest.TestCase):
    def setUp(self):
        self.background_vars = ["GENDER"]

        self.concepts = pd.DataFrame(
            {
                "subject_id": [
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "1",
                    "2",
                    "2",
                    "2",
                    "2",
                    "3",
                    "3",
                    "3",
                    "3",
                    "4",
                    "4",
                    "4",
                    "4",
                    "4",
                    "4",
                    "4",
                    "4",
                    "4",
                ],
                "time": [
                    datetime(2000, 5, 1),
                    NaT,
                    datetime(2015, 7, 1),
                    datetime(2015, 7, 2),
                    datetime(2016, 8, 2),
                    datetime(2022, 9, 3),
                    datetime(1995, 9, 3),
                    NaT,
                    datetime(2015, 9, 10),
                    datetime(2018, 4, 5),
                    datetime(1962, 11, 5),
                    NaT,
                    datetime(2020, 11, 6),
                    datetime(2022, 6, 6),
                    datetime(1975, 1, 6),
                    NaT,
                    datetime(2015, 12, 7),
                    datetime(2015, 12, 7),
                    datetime(2015, 12, 9),
                    datetime(2015, 12, 11),
                    datetime(2015, 12, 11),
                    datetime(2022, 10, 11),
                    datetime(2022, 10, 12),
                ],
                "code": [
                    "DOB",
                    "GENDER//F",
                    "A",
                    "B",
                    "C",
                    "DOD",
                    "DOB",
                    "GENDER//M",
                    "A",
                    "D",
                    "DOB",
                    "GENDER//M",
                    "D",
                    "D",
                    "DOB",
                    "GENDER//F",
                    "ADMISSION",
                    "E",
                    "A",
                    "B",
                    "DISCHARGE",
                    "X",
                    "Y",
                ],
            }
        )

        self.expected_segments = pd.Series(
            [0, 0, 1, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 1, 1, 1, 1, 2, 2],
            name="segment",
        )

        self.expected_patient_info = pd.DataFrame(
            {
                "subject_id": ["1", "2", "3", "4"],
                "birthdate": [
                    datetime(2000, 5, 1),
                    datetime(1995, 9, 3),
                    datetime(1962, 11, 5),
                    datetime(1975, 1, 6),
                ],
                "deathdate": [datetime(2022, 9, 3), NaT, NaT, NaT],
                "GENDER": ["F", "M", "M", "F"],
            }
        )

    def test_create_background(self):
        """
        Test the create_background function.
        """
        # Apply the function
        result, patient_info = create_background(self.concepts)

        # Expected number of rows: number of patients * number of background_vars
        expected_rows = len(self.concepts["code"])
        self.assertEqual(len(result), expected_rows)

        # Check that all background concepts are prefixed by 'BG_' and have birthdate as time
        self.assertEqual(result["code"].str.startswith("BG_").sum(), 4)
        bg_rows = result[result["code"].str.startswith("BG_")]
        dob_rows = result[result["code"] == "DOB"]
        self.assertTrue((bg_rows["time"] == dob_rows["time"].values).all())

        # Check patient_info
        self.assertTrue(
            (len(patient_info) == len(self.concepts["subject_id"].unique()))
        )
        self.assertTrue(
            all(
                patient_info.columns
                == ["subject_id", "birthdate", "deathdate", "GENDER"]
            )
        )
        pd.testing.assert_frame_equal(patient_info, self.expected_patient_info)

    def test_create_age_in_years(self):
        """
        Test the create_age_in_years function.
        """
        concepts_with_birthdate = self.concepts.merge(
            self.expected_patient_info[["subject_id", "birthdate"]],
            on="subject_id",
            how="left",
        )

        # Merge concepts with patients_info to get 'BIRTHDATE'
        result = create_age_in_years(concepts_with_birthdate)

        # Expected ages
        expected_ages = (
            self.concepts["time"]
            - self.expected_patient_info.set_index("subject_id")
            .loc[self.concepts["subject_id"]]["birthdate"]
            .values
        ).dt.days // 365.25

        # Assert the ages are as expected
        self.assertTrue((result["age"].dropna() == expected_ages.dropna()).all())

    def test_create_abspos(self):
        """
        Test the create_abspos function.
        """
        # Apply the function
        concepts_no_nan = self.concepts.dropna(
            subset=["time"]
        )  # Remove rows with NaT time (BG)
        result = create_abspos(concepts_no_nan)

        # Expected abspos
        expected_abspos = concepts_no_nan["time"].astype("int64") // 10**6 / 3600

        # Assert the abspos values are as expected
        self.assertTrue((result["abspos"] == expected_abspos).all())

    def test_create_segments(self):
        """
        Test the create_segments function.
        """
        # Prepare concepts DataFrame by adding 'abspos' (required for sorting) and 'BG' rows
        # concepts = self.concepts.rename(columns={"CONCEPT": "concept"})
        concepts_with_bg, _ = create_background(self.concepts)
        concepts_with_abspos = create_abspos(concepts_with_bg)

        # Apply the function
        sorted_concepts = sort_features(concepts_with_abspos)
        result = create_segments(sorted_concepts)

        # Assert the segments are as expected
        self.assertTrue((result["segment"] == self.expected_segments).all())


if __name__ == "__main__":
    unittest.main()
