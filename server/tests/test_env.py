import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from env import (
    minimax_h3_decode_offload,
    minimax_h3_disk_offload_cache_root,
    minimax_h3_lora_cache_root,
    minimax_h3_quantized_cache_root,
    video_generation_cache_root,
    _path,
    _positive_int,
)


class TestPositiveInt(TestCase):
    def test_valid_positive_integer(self) -> None:
        """Test parsing valid positive integers."""
        result = _positive_int("TEST_VAR", 10)
        assert result == 10

    def test_environment_variable_override(self) -> None:
        """Test that environment variable overrides default."""
        os.environ["TEST_POSITIVE_VAR"] = "42"
        try:
            result = _positive_int("TEST_POSITIVE_VAR", 10)
            assert result == 42
        finally:
            del os.environ["TEST_POSITIVE_VAR"]

    def test_zero_raises_error(self) -> None:
        """Test that zero raises ValueError."""
        os.environ["TEST_ZERO_VAR"] = "0"
        try:
            with self.assertRaises(ValueError) as ctx:
                _positive_int("TEST_ZERO_VAR", 10)
            assert "must be positive" in str(ctx.exception)
        finally:
            del os.environ["TEST_ZERO_VAR"]

    def test_negative_raises_error(self) -> None:
        """Test that negative values raise ValueError."""
        os.environ["TEST_NEGATIVE_VAR"] = "-5"
        try:
            with self.assertRaises(ValueError) as ctx:
                _positive_int("TEST_NEGATIVE_VAR", 10)
            assert "must be positive" in str(ctx.exception)
        finally:
            del os.environ["TEST_NEGATIVE_VAR"]

    def test_non_integer_raises_error(self) -> None:
        """Test that non-integer values raise ValueError."""
        os.environ["TEST_FLOAT_VAR"] = "3.14"
        try:
            with self.assertRaises(ValueError):
                _positive_int("TEST_FLOAT_VAR", 10)
        finally:
            del os.environ["TEST_FLOAT_VAR"]


class TestPath(TestCase):
    def test_default_path_when_not_set(self) -> None:
        """Test that default path is returned when env var not set."""
        default = Path("/default/path")
        result = _path("NONEXISTENT_PATH_VAR", default)
        assert result == default

    def test_environment_variable_override(self) -> None:
        """Test that environment variable path is used when set."""
        with TemporaryDirectory() as tmpdir:
            os.environ["TEST_PATH_VAR"] = tmpdir
            try:
                result = _path("TEST_PATH_VAR", Path("/default"))
                # Result should be resolved
                assert result.is_absolute()
                assert tmpdir in result.as_posix() or str(Path(tmpdir).resolve()) == str(result)
            finally:
                del os.environ["TEST_PATH_VAR"]

    def test_path_expansion(self) -> None:
        """Test that ~ in paths is expanded."""
        os.environ["TEST_HOME_PATH_VAR"] = "~/test_shrimply"
        try:
            result = _path("TEST_HOME_PATH_VAR", Path("/default"))
            # Should expand ~ to home directory and resolve
            assert result.is_absolute()
            assert "~" not in str(result)
            assert str(Path.home()) in str(result)
        finally:
            del os.environ["TEST_HOME_PATH_VAR"]

    def test_path_resolution(self) -> None:
        """Test that paths are resolved (made absolute)."""
        os.environ["TEST_REL_PATH_VAR"] = "./relative/path"
        try:
            result = _path("TEST_REL_PATH_VAR", Path("/default"))
            assert result.is_absolute()
        finally:
            del os.environ["TEST_REL_PATH_VAR"]


class TestCachePaths(TestCase):
    """Test cache root path functions."""

    def test_video_generation_cache_root_default(self) -> None:
        """Test default video generation cache root."""
        # Remove env var if set
        old_value = os.environ.pop("SHRIMPLY_VIDEO_GENERATION_CACHE", None)
        try:
            result = video_generation_cache_root()
            assert result.is_absolute()
            assert "video-generation-cache" in result.as_posix()
        finally:
            if old_value:
                os.environ["SHRIMPLY_VIDEO_GENERATION_CACHE"] = old_value

    def test_minimax_h3_quantized_cache_root_default(self) -> None:
        """Test default minimax quantized cache root."""
        old_value = os.environ.pop("MINIMAX_H3_QUANTIZED_CACHE", None)
        try:
            result = minimax_h3_quantized_cache_root()
            assert result.is_absolute()
            assert "quantized-cache" in result.as_posix()
        finally:
            if old_value:
                os.environ["MINIMAX_H3_QUANTIZED_CACHE"] = old_value

    def test_minimax_h3_disk_offload_cache_root_default(self) -> None:
        """Test default minimax disk offload cache root."""
        old_value = os.environ.pop("MINIMAX_H3_DISK_OFFLOAD_CACHE", None)
        try:
            result = minimax_h3_disk_offload_cache_root()
            assert result.is_absolute()
            assert "disk-offload-cache" in result.as_posix()
        finally:
            if old_value:
                os.environ["MINIMAX_H3_DISK_OFFLOAD_CACHE"] = old_value

    def test_minimax_h3_lora_cache_root_default(self) -> None:
        """Test default minimax lora cache root."""
        old_value = os.environ.pop("MINIMAX_H3_LORA_CACHE", None)
        try:
            result = minimax_h3_lora_cache_root()
            assert result.is_absolute()
            assert "lora-cache" in result.as_posix()
        finally:
            if old_value:
                os.environ["MINIMAX_H3_LORA_CACHE"] = old_value


class TestMinimax3DecodeOffload(TestCase):
    """Test minimax H3 decode offload configuration."""

    def test_default_is_auto(self) -> None:
        """Test default decode offload value is 'auto'."""
        old_value = os.environ.pop("MINIMAX_H3_DECODE_OFFLOAD", None)
        try:
            result = minimax_h3_decode_offload()
            assert result == "auto"
        finally:
            if old_value:
                os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = old_value

    def test_valid_values(self) -> None:
        """Test that valid offload values are accepted."""
        for value in ["auto", "gpu", "disk"]:
            os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = value
            try:
                result = minimax_h3_decode_offload()
                assert result == value
            finally:
                del os.environ["MINIMAX_H3_DECODE_OFFLOAD"]

    def test_case_insensitive(self) -> None:
        """Test that value matching is case-insensitive."""
        os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = "GPU"
        try:
            result = minimax_h3_decode_offload()
            assert result == "gpu"
        finally:
            del os.environ["MINIMAX_H3_DECODE_OFFLOAD"]

    def test_invalid_value_raises_error(self) -> None:
        """Test that invalid values raise ValueError."""
        os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = "invalid"
        try:
            with self.assertRaises(ValueError) as ctx:
                minimax_h3_decode_offload()
            assert "must be auto, gpu, or disk" in str(ctx.exception)
        finally:
            del os.environ["MINIMAX_H3_DECODE_OFFLOAD"]
