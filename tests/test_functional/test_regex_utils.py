import unittest
import pandas as pd
from bonsai.functional.regex_utils import is_valid_regex, filter_rows_by_regex


class TestRegexUtils(unittest.TestCase):
    def test_is_valid_regex(self):
        self.assertTrue(is_valid_regex(r"^abc$"))
        self.assertFalse(is_valid_regex(r"[unclosed"))

    def test_filter_rows_by_regex(self):
        df = pd.DataFrame({"col": ["foo", "bar", "baz"]})
        filtered = filter_rows_by_regex(df, "col", r"ba.")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["col"], "foo")
