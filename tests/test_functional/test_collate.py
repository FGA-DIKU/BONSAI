import unittest
import torch
from bonsai.functional.collate import dynamic_padding


class TestCollate(unittest.TestCase):
    def setUp(self):
        self.batch = [
            {
                "code": torch.tensor([1, 2]),
                "abspos": torch.tensor([0, 1]),
                "age": torch.tensor([10, 20]),
                "segment": torch.tensor([0, 0]),
                "attention_mask": torch.tensor([1, 1]),
                "target": torch.tensor([0, 1]),
                "subject_id": 1,
            },
            {
                "code": torch.tensor([3]),
                "abspos": torch.tensor([2]),
                "age": torch.tensor([30]),
                "segment": torch.tensor([1]),
                "attention_mask": torch.tensor([1]),
                "target": torch.tensor([1]),
                "subject_id": 2,
            },
        ]

    def test_dynamic_padding_shapes(self):
        output = dynamic_padding(self.batch)
        self.assertEqual(output["code"].shape, (2, 2))
        self.assertEqual(output["abspos"].shape, (2, 2))
        self.assertEqual(output["age"].shape, (2, 2))
        self.assertEqual(output["segment"].shape, (2, 2))
        self.assertEqual(output["attention_mask"].shape, (2, 2))
        self.assertEqual(output["target"].shape, (2, 2))
        self.assertEqual(output["subject_id"].shape, (2,))

    def test_dynamic_padding_values(self):
        output = dynamic_padding(self.batch)
        # Check padding value for target
        self.assertEqual(output["code"][1, 1].item(), 0)
        self.assertEqual(output["abspos"][1, 1].item(), 0)
        self.assertEqual(output["age"][1, 1].item(), 0)
        self.assertEqual(output["segment"][1, 1].item(), 0)
        self.assertEqual(output["attention_mask"][1, 1].item(), 0)
        self.assertEqual(output["target"][1, 1].item(), -100)
