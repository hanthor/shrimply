from unittest import TestCase
from unittest.mock import MagicMock, patch

from api.gpu import (
    bf16_supported,
    is_out_of_memory,
    select_device,
)


class TestBf16Supported(TestCase):
    """Test BF16 support detection."""

    def test_bf16_supported_on_cuda_device(self) -> None:
        """Test BF16 support detection on CUDA device."""
        with patch("api.gpu.device", "cuda:0"):
            with patch("torch.cuda.device"):
                with patch("torch.cuda.is_bf16_supported", return_value=True):
                    result = bf16_supported()
                    assert result is True

    def test_bf16_not_supported_on_cuda_device(self) -> None:
        """Test when BF16 is not supported on CUDA device."""
        with patch("api.gpu.device", "cuda:0"):
            with patch("torch.cuda.device"):
                with patch("torch.cuda.is_bf16_supported", return_value=False):
                    result = bf16_supported()
                    assert result is False

    def test_bf16_not_supported_on_cpu(self) -> None:
        """Test that BF16 is not supported on CPU device."""
        with patch("api.gpu.device", "cpu"):
            result = bf16_supported()
            assert result is False


class TestIsOutOfMemory(TestCase):
    """Test out-of-memory error detection."""

    def test_out_of_memory_lowercase(self) -> None:
        """Test detection of 'out of memory' error message."""
        assert is_out_of_memory("CUDA out of memory error") is True
        assert is_out_of_memory("out of memory") is True

    def test_not_enough_gpu_memory(self) -> None:
        """Test detection of 'not enough gpu memory' error message."""
        assert is_out_of_memory("not enough gpu memory") is True
        assert is_out_of_memory("Not enough GPU memory available") is True

    def test_case_insensitive_matching(self) -> None:
        """Test that matching is case-insensitive."""
        assert is_out_of_memory("OUT OF MEMORY") is True
        assert is_out_of_memory("NOT ENOUGH GPU MEMORY") is True
        assert is_out_of_memory("Out Of Memory") is True

    def test_non_oom_errors(self) -> None:
        """Test that non-OOM errors are not detected."""
        assert is_out_of_memory("RuntimeError: invalid device") is False
        assert is_out_of_memory("ValueError: invalid argument") is False
        assert is_out_of_memory("") is False

    def test_partial_matches_not_detected(self) -> None:
        """Test that partial matches are not false positives."""
        assert is_out_of_memory("memory is available") is False
        assert is_out_of_memory("out on a limb") is False


class TestSelectDevice(TestCase):
    """Test compute device selection."""

    @patch("api.gpu.torch.cuda.device_count", return_value=2)
    @patch("api.resource.scheduler.select_device")
    def test_select_cpu_device(
        self, mock_select: MagicMock, mock_device_count: MagicMock
    ) -> None:
        """Test selecting CPU device."""
        select_device("cpu")
        mock_select.assert_called_once_with("cpu")

    @patch("api.gpu.torch.cuda.device_count", return_value=2)
    @patch("api.resource.scheduler.select_device")
    def test_select_cuda_device(
        self, mock_select: MagicMock, mock_device_count: MagicMock
    ) -> None:
        """Test selecting CUDA device."""
        select_device("cuda:0")
        mock_select.assert_called_once_with("cuda:0")

    @patch("api.gpu.torch.cuda.device_count", return_value=2)
    @patch("api.resource.scheduler.select_device")
    def test_select_multiple_cuda_devices(
        self, mock_select: MagicMock, mock_device_count: MagicMock
    ) -> None:
        """Test selecting multiple CUDA devices."""
        for i in range(2):
            select_device(f"cuda:{i}")
            mock_select.assert_called_with(f"cuda:{i}")

    @patch("api.gpu.torch.cuda.device_count", return_value=1)
    def test_unavailable_cuda_device_raises_error(
        self, mock_device_count: MagicMock
    ) -> None:
        """Test that selecting unavailable CUDA device raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            select_device("cuda:5")
        assert "unavailable" in str(ctx.exception).lower()
        assert "cuda:5" in str(ctx.exception)

    @patch("api.gpu.torch.cuda.device_count", return_value=2)
    def test_invalid_device_format_raises_error(
        self, mock_device_count: MagicMock
    ) -> None:
        """Test that invalid device format raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            select_device("gpu:0")
        assert "unavailable" in str(ctx.exception).lower()

    @patch("api.gpu.torch.cuda.device_count", return_value=0)
    def test_cuda_device_unavailable_when_no_gpus(
        self, mock_device_count: MagicMock
    ) -> None:
        """Test that CUDA device is unavailable when no GPUs present."""
        with self.assertRaises(ValueError):
            select_device("cuda:0")
