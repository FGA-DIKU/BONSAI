import unittest
from unittest.mock import patch, MagicMock
from bonsai.functional.pathing import get_experiment_output_path


class TestPathing(unittest.TestCase):
    @patch("bonsai.functional.pathing.HydraConfig")
    def test_get_experiment_output_path(self, mock_hydra):
        mock_runtime = MagicMock()
        mock_runtime.output_dir = "/tmp/experiment"
        mock_hydra.get.return_value.runtime = mock_runtime
        result = get_experiment_output_path()
        self.assertEqual(result, "/tmp/experiment")
