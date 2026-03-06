import unittest
import torch
from bonsai.functional.censoring import censor_subject, append_predict_token


class TestCensoring(unittest.TestCase):
    def setUp(self) -> None:
        self.subject = {
            "code": torch.tensor([1, 2, 3, 4, 5, 6]),
            "abspos": torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.float),
            "segment": torch.tensor([0, 0, 1, 1, 2, 2]),
            "age": torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.float),
        }

    def test_censor_subject_truncate(self):
        subj = {k: v.clone() for k, v in self.subject.items()}
        # Censor at abspos=35, should keep first 3 tokens (10,20,30)
        censored = censor_subject(subj, censor_date_abspos=35)
        self.assertTrue(torch.equal(censored["code"], torch.tensor([1, 2, 3])))
        self.assertTrue(
            torch.equal(
                censored["abspos"], torch.tensor([10, 20, 30], dtype=torch.float)
            )
        )
        self.assertTrue(torch.equal(censored["segment"], torch.tensor([0, 0, 1])))
        self.assertTrue(
            torch.equal(censored["age"], torch.tensor([5, 6, 7], dtype=torch.float))
        )

    def test_censor_subject_no_truncate(self):
        subj = {k: v.clone() for k, v in self.subject.items()}
        # Censor at abspos=100, should keep all
        censored = censor_subject(subj, censor_date_abspos=100)
        for k in self.subject:
            self.assertTrue(torch.equal(censored[k], self.subject[k]))

    def test_censor_subject_with_predict_token(self):
        subj = {k: v.clone() for k, v in self.subject.items()}
        predict_token_id = 99
        censor_date = 35
        censored = censor_subject(
            subj, censor_date_abspos=censor_date, predict_token_id=predict_token_id
        )
        # Should keep first 3, then append predict token
        self.assertTrue(torch.equal(censored["code"], torch.tensor([1, 2, 3, 99])))
        self.assertTrue(
            torch.equal(
                censored["abspos"], torch.tensor([10, 20, 30, 35], dtype=torch.float)
            )
        )
        self.assertTrue(torch.equal(censored["segment"], torch.tensor([0, 0, 1, 2])))
        # Check age: last value is calculated as in append_predict_token
        expected_age = float((censor_date - 10) / (365.25 * 24))
        self.assertAlmostEqual(censored["age"][-1].item(), expected_age, places=6)

    def test_append_predict_token(self):
        subj = {k: v.clone() for k, v in self.subject.items()}
        predict_token_id = 42
        censor_date = 77
        appended = append_predict_token(subj, censor_date, predict_token_id)
        self.assertTrue(
            torch.equal(appended["code"], torch.tensor([1, 2, 3, 4, 5, 6, 42]))
        )
        self.assertTrue(
            torch.equal(
                appended["abspos"],
                torch.tensor([10, 20, 30, 40, 50, 60, 77], dtype=torch.float),
            )
        )
        self.assertTrue(
            torch.equal(appended["segment"], torch.tensor([0, 0, 1, 1, 2, 2, 3]))
        )
        expected_age = float((censor_date - 10) / (365.25 * 24))
        self.assertAlmostEqual(appended["age"][-1].item(), expected_age, places=6)
