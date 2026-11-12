import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from bonsai.functional.pathing import check_run_dir_unused, get_experiment_output_path


class TestPathing(unittest.TestCase):
    @patch("bonsai.functional.pathing.HydraConfig")
    def test_get_experiment_output_path(self, mock_hydra):
        mock_runtime = MagicMock()
        mock_runtime.output_dir = "/tmp/experiment"
        mock_hydra.get.return_value.runtime = mock_runtime
        result = get_experiment_output_path()
        self.assertEqual(result, "/tmp/experiment")

    def test_check_run_dir_unused(self):
        with tempfile.TemporaryDirectory() as run_dir:
            check_run_dir_unused(run_dir, None)  # Fresh folder

            Path(run_dir, "best.ckpt").touch()
            check_run_dir_unused(run_dir, f"{run_dir}/last.ckpt")  # Resume
            with self.assertRaises(FileExistsError):
                check_run_dir_unused(run_dir, None)  # Reused run_id
