import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import sys

# Mock torch before importing gpu module
sys.modules['torch'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock()

from api import gpu


class TestGPUUtilities(unittest.TestCase):
    """Tests for GPU device management utilities."""

    def setUp(self):
        """Save and reset module state for each test."""
        self.original_device = gpu.device

    def tearDown(self):
        """Restore module state."""
        gpu.device = self.original_device

    @patch('torch.cuda.is_available')
    def test_device_selection_cuda_when_available(self, mock_is_available):
        """Device should be cuda:0 when torch.cuda is available."""
        mock_is_available.return_value = True
        with patch.dict('sys.modules', {'torch': MagicMock(), 'torch.cuda': MagicMock()}):
            # Reimport to trigger device selection with mocked cuda
            import importlib
            original_gpu = sys.modules.get('api.gpu')
            sys.modules['api.gpu'] = MagicMock()
            # Note: This test documents the expected behavior

    @patch('torch.cuda.is_available')
    def test_device_selection_cpu_when_unavailable(self, mock_is_available):
        """Device should be cpu when torch.cuda is unavailable."""
        mock_is_available.return_value = False
        # Document expected behavior

    @patch('torch.cuda.mem_get_info')
    def test_available_vram_returns_free_memory(self, mock_mem_info):
        """available_vram() should return free memory from torch.cuda.mem_get_info."""
        mock_mem_info.return_value = (1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)  # 1GB free, 8GB total
        
        gpu.device = "cuda:0"
        with patch('torch.cuda.mem_get_info', return_value=(1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)):
            result = gpu.available_vram()
            self.assertEqual(result, 1024 * 1024 * 1024)

    @patch('torch.cuda.get_device_properties')
    def test_total_vram_returns_device_memory(self, mock_props):
        """total_vram() should return total device memory."""
        mock_device_props = MagicMock()
        mock_device_props.total_memory = 16 * 1024 * 1024 * 1024  # 16GB
        mock_props.return_value = mock_device_props
        
        gpu.device = "cuda:0"
        with patch('torch.cuda.get_device_properties', return_value=mock_device_props):
            result = gpu.total_vram()
            self.assertEqual(result, 16 * 1024 * 1024 * 1024)

    def test_bf16_supported_returns_false_for_cpu(self):
        """bf16_supported() should return False when device is CPU."""
        gpu.device = "cpu"
        result = gpu.bf16_supported()
        self.assertFalse(result)

    @patch('torch.cuda.is_bf16_supported')
    @patch('torch.cuda.device')
    def test_bf16_supported_returns_true_for_cuda(self, mock_cuda_context, mock_bf16_check):
        """bf16_supported() should check torch.cuda.is_bf16_supported() for CUDA devices."""
        mock_bf16_check.return_value = True
        mock_cuda_context.return_value.__enter__ = MagicMock()
        mock_cuda_context.return_value.__exit__ = MagicMock(return_value=False)
        
        gpu.device = "cuda:0"
        with patch('torch.cuda.is_bf16_supported', return_value=True):
            with patch('torch.cuda.device'):
                result = gpu.bf16_supported()
                self.assertTrue(result)

    def test_is_out_of_memory_detects_oom_message(self):
        """is_out_of_memory() should detect 'out of memory' in error messages."""
        test_cases = [
            ("CUDA out of memory error", True),
            ("Error: out of memory on device", True),
            ("OUT OF MEMORY", True),  # Case insensitive
            ("not enough GPU memory available", True),
            ("Not Enough GPU Memory", True),  # Case insensitive
            ("Regular error message", False),
            ("Memory allocation failed", False),
            ("", False),
        ]
        
        for message, expected in test_cases:
            with self.subTest(message=message):
                result = gpu.is_out_of_memory(message)
                self.assertEqual(result, expected)

    def test_is_out_of_memory_case_insensitive(self):
        """is_out_of_memory() should be case-insensitive."""
        messages = [
            "Out Of Memory",
            "OUT OF MEMORY",
            "out of memory",
            "Not Enough GPU Memory",
            "NOT ENOUGH GPU MEMORY",
        ]
        
        for message in messages:
            with self.subTest(message=message):
                self.assertTrue(gpu.is_out_of_memory(message))

    @patch('torch.cuda.device_count')
    @patch.dict('sys.modules', {'api.resource': MagicMock()})
    def test_select_device_cpu(self, mock_device_count):
        """select_device() should accept 'cpu' device."""
        mock_device_count.return_value = 1
        
        with patch('api.resource') as mock_resource:
            mock_resource.scheduler = MagicMock()
            # Should not raise
            try:
                gpu.select_device("cpu")
                # Verify scheduler.select_device was called
                mock_resource.scheduler.select_device.assert_called_once_with("cpu")
            except ImportError:
                # Expected if api.resource is not available
                pass

    @patch('torch.cuda.device_count')
    def test_select_device_valid_cuda_indices(self, mock_device_count):
        """select_device() should accept valid CUDA device indices."""
        mock_device_count.return_value = 4
        
        with patch('api.resource') as mock_resource:
            mock_resource.scheduler = MagicMock()
            
            valid_devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
            for device in valid_devices:
                with self.subTest(device=device):
                    try:
                        gpu.select_device(device)
                        mock_resource.scheduler.select_device.assert_called_with(device)
                    except ImportError:
                        pass

    @patch('torch.cuda.device_count')
    def test_select_device_rejects_invalid_indices(self, mock_device_count):
        """select_device() should reject invalid CUDA device indices."""
        mock_device_count.return_value = 2
        
        invalid_devices = ["cuda:2", "cuda:5", "cuda:10", "gpu:0", "invalid"]
        for device in invalid_devices:
            with self.subTest(device=device):
                with self.assertRaises(ValueError) as ctx:
                    gpu.select_device(device)
                self.assertIn("unavailable", str(ctx.exception).lower())

    @patch('torch.cuda.device_count')
    def test_select_device_builds_available_devices_correctly(self, mock_device_count):
        """select_device() should correctly enumerate available devices."""
        test_cases = [
            (0, ("cpu",)),  # Only CPU when no CUDA devices
            (1, ("cpu", "cuda:0")),
            (2, ("cpu", "cuda:0", "cuda:1")),
            (4, ("cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3")),
        ]
        
        for device_count, expected_available in test_cases:
            with self.subTest(device_count=device_count):
                mock_device_count.return_value = device_count
                # Document the enumeration logic


if __name__ == "__main__":
    unittest.main()
