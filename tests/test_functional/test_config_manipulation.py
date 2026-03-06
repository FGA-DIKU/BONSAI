import unittest
from omegaconf import OmegaConf
from bonsai.functional.config_manipulation import merge_configs_and_drop_duplicate_keys


class TestConfigManipulation(unittest.TestCase):
    def test_merge_and_drop(self):
        pretrain_cfg = {"a": 1, "b": 2, "vocab_size": 100, "pad_token_id": 0}
        finetune_cfg = OmegaConf.create(
            {"a": 1, "b": 2, "c": 3, "cls_token_id": 1, "sep_token_id": 2}
        )
        merged = merge_configs_and_drop_duplicate_keys(pretrain_cfg, finetune_cfg)
        self.assertNotIn("vocab_size", merged)
        self.assertNotIn("pad_token_id", merged)
        self.assertNotIn("cls_token_id", merged)
        self.assertNotIn("sep_token_id", merged)
        self.assertEqual(merged["a"], 1)
        self.assertEqual(merged["b"], 2)
        self.assertEqual(merged["c"], 3)

    def test_conflict_raises(self):
        pretrain_cfg = {"a": 1}
        finetune_cfg = {"a": 2}
        with self.assertRaises(ValueError):
            merge_configs_and_drop_duplicate_keys(pretrain_cfg, finetune_cfg)
