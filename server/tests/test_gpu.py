from unittest import TestCase
from unittest.mock import MagicMock, patch

from api.gpu import (
    available_vram,
    bf16_supported,
    is_out_of_memory,
    select_device,
    total_vram,
)


class GPUUtilityTests(TestCase):
    """Unit tests for gpu.py device management utilities."""

    def test_bf16_supported_cpu(self) -> None:
        """bf16_supported returns False on CPU."""
        with patch("api.gpu.device", "cpu"):
            self.assertFalse(bf16_supported())

    def test_bf16_supported_cuda(self) -> None:
        """bf16_supported checks CUDA capabilities when on GPU."""
        with patch("api.gpu.device", "cuda:0"), patch(
            "torch.cuda.is_bf16_supported", return_value=True
        ):
            self.assertTrue(bf16_supported())

    def test_bf16_supported_cuda_not_supported(self) -> None:
        """bf16_supported returns False when CUDA doesn't support BF16."""
        with patch("api.gpu.device", "cuda:0"), patch(
            "torch.cuda.is_bf16_supported", return_value=False
        ):
            self.assertFalse(bf16_supported())

    def test_is_out_of_memory_exact_match(self) -> None:
        """is_out_of_memory detects exact 'out of memory' message."""
        self.assertTrue(is_out_of_memory("RuntimeError: out of memory"))

    def test_is_out_of_memory_case_insensitive(self) -> None:
        """is_out_of_memory is case-insensitive."""
        self.assertTrue(is_out_of_memory("RuntimeError: OUT OF MEMORY"))
        self.assertTrue(is_out_of_memory("RuntimeError: Out Of Memory"))

    def test_is_out_of_memory_gpu_memory_variant(self) -> None:
        """is_out_of_memory detects 'not enough gpu memory' variant."""
        self.assertTrue(is_out_of_memory("RuntimeError: not enough gpu memory"))
        self.assertTrue(is_out_of_memory("RuntimeError: NOT ENOUGH GPU MEMORY"))

    def test_is_out_of_memory_false_for_other_errors(self) -> None:
        """is_out_of_memory returns False for unrelated error messages."""
        self.assertFalse(is_out_of_memory("RuntimeError: device not found"))
        self.assertFalse(is_out_of_memory("RuntimeError: invalid memory address"))
        self.assertFalse(is_out_of_memory(""))

    def test_is_out_of_memory_partial_match(self) -> None:
        """is_out_of_memory detects OOM in longer messages."""
        self.assertTrue(
            is_out_of_memory(
                "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.62 GiB total capacity;"
            )
        )

    def test_available_vram_calls_torch_cuda(self) -> None:
        """available_vram calls torch.cuda.mem_get_info."""
        with patch("torch.cuda.mem_get_info", return_value=(1024, 2048)):
            result = available_vram()
            self.assertEqual(result, 1024)

    def test_total_vram_calls_torch_cuda(self) -> None:
        """total_vram calls torch.cuda.get_device_properties."""
        mock_props = MagicMock()
        mock_props.total_memory = 2048
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            result = total_vram()
            self.assertEqual(result, 2048)

    def test_select_device_valid_cpu(self) -> None:
        """select_device accepts 'cpu' as a valid device."""
        with patch("api.gpu.torch.cuda.device_count", return_value=1), patch(
            "api.resource.scheduler.select_device"
        ) as mock_select:
            select_device("cpu")
            mock_select.assert_called_once_with("cpu")

    def test_select_device_valid_cuda(self) -> None:
        """select_device accepts valid CUDA devices."""
        with patch("api.gpu.torch.cuda.device_count", return_value=2), patch(
            "api.resource.scheduler.select_device"
        ) as mock_select:
            select_device("cuda:0")
            mock_select.assert_called_once_with("cuda:0")
            mock_select.reset_mock()

            select_device("cuda:1")
            mock_select.assert_called_once_with("cuda:1")

    def test_select_device_invalid_index(self) -> None:
        """select_device raises ValueError for out-of-range CUDA device."""
        with patch("api.gpu.torch.cuda.device_count", return_value=1), patch(
            "api.resource.scheduler.select_device"
        ):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                select_device("cuda:2")

    def test_select_device_invalid_format(self) -> None:
        """select_device raises ValueError for malformed device string."""
        with patch("api.gpu.torch.cuda.device_count", return_value=1), patch(
            "api.resource.scheduler.select_device"
        ):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                select_device("gpu:0")

    def test_select_device_error_message_includes_device_name(self) -> None:
        """select_device error message includes the rejected device name."""
        with patch("api.gpu.torch.cuda.device_count", return_value=0), patch(
            "api.resource.scheduler.select_device"
        ):
            try:
                select_device("cuda:5")
                self.fail("Expected ValueError")
            except ValueError as e:
                self.assertIn("cuda:5", str(e))
