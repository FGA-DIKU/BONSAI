import unittest
import torch
from bonsai.functional.truncation import truncate_subject


class TestTruncation(unittest.TestCase):
    def setUp(self) -> None:
        self.subject = {
            "subject_id": 10,
            "code": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "abspos": torch.tensor([0, 0, 0, 1, 1, 2, 3, 4, 5, 6]),
            "age": torch.tensor([0, 0, 0, 10, 10, 20, 30, 40, 50, 60]),
            "segment": torch.tensor([0, 0, 0, 1, 1, 2, 3, 4, 5, 6]),
        }

    def test_no_truncate(self):
        post_subject = truncate_subject(self.subject, max_len=100, background_length=3)
        self.assertDictEqual(self.subject, post_subject)

    def test_truncate(self):
        post_subject = truncate_subject(self.subject, max_len=5, background_length=3)
        expected_subject = {
            "subject_id": self.subject["subject_id"],
            "code": torch.tensor([1, 2, 3, 9, 10]),
            "abspos": torch.tensor([0, 0, 0, 5, 6]),
            "age": torch.tensor([0, 0, 0, 50, 60]),
            "segment": torch.tensor([0, 0, 0, 5, 6]),
        }
        self.assertEqual(
            post_subject.pop("subject_id"), expected_subject.pop("subject_id")
        )
        for key in post_subject:
            torch.testing.assert_close(post_subject[key], expected_subject[key])

    def test_equal_length(self):
        post_subject = truncate_subject(self.subject, max_len=7, background_length=3)
        self.assertDictEqual(self.subject, post_subject)
