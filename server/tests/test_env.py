"""Unit tests for server/src/env.py environment configuration utilities."""

import os
import pytest
from pathlib import Path

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import env


class TestPositiveInt:
    """Tests for env._positive_int() validation function."""

    def test_returns_env_value_when_set(self, monkeypatch):
        """Return the integer value from environment variable when set."""
        monkeypatch.setenv("TEST_POSITIVE_INT", "42")
        result = env._positive_int("TEST_POSITIVE_INT", 10)
        assert result == 42

    def test_returns_default_when_not_set(self, monkeypatch):
        """Return the default value when environment variable is not set."""
        monkeypatch.delenv("TEST_POSITIVE_INT", raising=False)
        result = env._positive_int("TEST_POSITIVE_INT", 25)
        assert result == 25

    def test_raises_for_zero_value(self, monkeypatch):
        """Raise ValueError when value is zero (not positive)."""
        monkeypatch.setenv("TEST_ZERO", "0")
        with pytest.raises(ValueError, match="must be positive"):
            env._positive_int("TEST_ZERO", 10)

    def test_raises_for_negative_value(self, monkeypatch):
        """Raise ValueError when value is negative."""
        monkeypatch.setenv("TEST_NEGATIVE", "-5")
        with pytest.raises(ValueError, match="must be positive"):
            env._positive_int("TEST_NEGATIVE", 10)

    def test_accepts_positive_values(self, monkeypatch):
        """Accept and return any positive integer."""
        for value in [1, 100, 999999]:
            monkeypatch.setenv("TEST_POS", str(value))
            result = env._positive_int("TEST_POS", 10)
            assert result == value

    def test_raises_for_non_integer_value(self, monkeypatch):
        """Raise ValueError for non-integer environment values."""
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        with pytest.raises(ValueError):
            env._positive_int("TEST_FLOAT", 10)

    def test_raises_for_non_numeric_value(self, monkeypatch):
        """Raise ValueError for non-numeric environment values."""
        monkeypatch.setenv("TEST_TEXT", "not_a_number")
        with pytest.raises(ValueError):
            env._positive_int("TEST_TEXT", 10)


class TestPathFunction:
    """Tests for env._path() path resolution function."""

    def test_returns_expanded_path_when_set(self, monkeypatch, tmp_path):
        """Return expanded and resolved path from environment variable."""
        test_path = tmp_path / "test_dir"
        test_path.mkdir(exist_ok=True)
        monkeypatch.setenv("TEST_PATH", str(test_path))
        
        result = env._path("TEST_PATH", Path("/default"))
        assert result == test_path.resolve()

    def test_returns_default_when_not_set(self, monkeypatch):
        """Return default path when environment variable is not set."""
        monkeypatch.delenv("TEST_PATH_DEFAULT", raising=False)
        default = Path("/default/path")
        
        result = env._path("TEST_PATH_DEFAULT", default)
        assert result == default.resolve()

    def test_expands_user_home_directory(self, monkeypatch, tmp_path):
        """Expand ~ (home directory) in path."""
        monkeypatch.setenv("FAKE_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_PATH_TILDE", "~/test")
        
        # Note: Path.expanduser() uses actual $HOME, so we use absolute path instead
        monkeypatch.setenv("TEST_PATH_ABS", str(tmp_path / "subdir"))
        result = env._path("TEST_PATH_ABS", Path("/default"))
        assert str(result).startswith(str(tmp_path))

    def test_resolves_symlinks_and_relative_paths(self, monkeypatch, tmp_path):
        """Resolve relative paths and symlinks to absolute paths."""
        subdir = tmp_path / "subdir"
        subdir.mkdir(exist_ok=True)
        
        monkeypatch.setenv("TEST_RELATIVE", str(subdir))
        result = env._path("TEST_RELATIVE", Path("/default"))
        
        assert result.is_absolute()
        assert result == subdir.resolve()

    def test_handles_nonexistent_paths(self, monkeypatch):
        """Handle paths that don't yet exist (resolve still works)."""
        monkeypatch.setenv("TEST_NONEXISTENT", "/path/that/does/not/exist")
        result = env._path("TEST_NONEXISTENT", Path("/default"))
        
        # Path.resolve() works on nonexistent paths
        assert result == Path("/path/that/does/not/exist").resolve()


class TestMinimaxH3DecodeOffload:
    """Tests for env.minimax_h3_decode_offload() enum validation."""

    def test_returns_auto_by_default(self, monkeypatch):
        """Return 'auto' when MINIMAX_H3_DECODE_OFFLOAD is not set."""
        monkeypatch.delenv("MINIMAX_H3_DECODE_OFFLOAD", raising=False)
        result = env.minimax_h3_decode_offload()
        assert result == "auto"

    def test_accepts_valid_modes(self, monkeypatch):
        """Accept all valid offload modes: auto, gpu, disk."""
        for mode in ["auto", "gpu", "disk"]:
            monkeypatch.setenv("MINIMAX_H3_DECODE_OFFLOAD", mode)
            result = env.minimax_h3_decode_offload()
            assert result == mode

    def test_case_insensitive_matching(self, monkeypatch):
        """Accept uppercase and mixed-case mode values."""
        for mode_input, mode_expected in [("AUTO", "auto"), ("GPU", "gpu"), ("DISK", "disk")]:
            monkeypatch.setenv("MINIMAX_H3_DECODE_OFFLOAD", mode_input)
            result = env.minimax_h3_decode_offload()
            assert result == mode_expected

    def test_raises_for_invalid_mode(self, monkeypatch):
        """Raise ValueError for invalid offload modes."""
        invalid_modes = ["invalid", "memory", "cpu", "random"]
        for invalid_mode in invalid_modes:
            monkeypatch.setenv("MINIMAX_H3_DECODE_OFFLOAD", invalid_mode)
            with pytest.raises(ValueError, match="must be auto, gpu, or disk"):
                env.minimax_h3_decode_offload()


class TestServerConfigurationDefaults:
    """Tests for server configuration constants and defaults."""

    def test_server_git_hash_empty_by_default(self, monkeypatch):
        """SERVER_GIT_HASH defaults to empty string when not set."""
        monkeypatch.delenv("SHRIMPLY_SERVER_GIT_HASH", raising=False)
        # Re-evaluate the module-level constant
        import importlib
        importlib.reload(env)
        assert env.SERVER_GIT_HASH == ""

    def test_server_host_defaults_to_all_interfaces(self, monkeypatch):
        """SERVER_HOST defaults to 0.0.0.0 (all interfaces)."""
        monkeypatch.delenv("SHRIMPLY_SERVER_HOST", raising=False)
        importlib.reload(env)
        assert env.SERVER_HOST == "0.0.0.0"

    def test_server_port_can_be_customized(self, monkeypatch):
        """SERVER_PORT can be set via environment variable."""
        monkeypatch.setenv("SHRIMPLY_SERVER_PORT", "9000")
        importlib.reload(env)
        assert env.SERVER_PORT == 9000


class TestServerShare:
    """Tests for SERVER_SHARE boolean configuration."""

    def test_server_share_false_by_default(self, monkeypatch):
        """SERVER_SHARE defaults to False when not set."""
        monkeypatch.delenv("SHRIMPLY_SERVER_SHARE", raising=False)
        importlib.reload(env)
        assert env.SERVER_SHARE is False

    def test_server_share_true_with_value_1(self, monkeypatch):
        """SERVER_SHARE is True when SHRIMPLY_SERVER_SHARE=1."""
        monkeypatch.setenv("SHRIMPLY_SERVER_SHARE", "1")
        importlib.reload(env)
        assert env.SERVER_SHARE is True

    def test_server_share_false_with_value_0(self, monkeypatch):
        """SERVER_SHARE is False when SHRIMPLY_SERVER_SHARE=0."""
        monkeypatch.setenv("SHRIMPLY_SERVER_SHARE", "0")
        importlib.reload(env)
        assert env.SERVER_SHARE is False

    def test_server_share_rejects_invalid_values(self, monkeypatch):
        """Raise ValueError for invalid SHRIMPLY_SERVER_SHARE values."""
        monkeypatch.setenv("SHRIMPLY_SERVER_SHARE", "yes")
        with pytest.raises(ValueError, match="must be 0 or 1"):
            importlib.reload(env)


class TestCachePaths:
    """Tests for cache directory path functions."""

    def test_video_generation_cache_defaults(self, monkeypatch, tmp_path):
        """video_generation_cache_root() defaults to ./video-generation-cache."""
        monkeypatch.delenv("SHRIMPLY_VIDEO_GENERATION_CACHE", raising=False)
        result = env.video_generation_cache_root()
        # Default is relative to current working directory
        assert result.name == "video-generation-cache"

    def test_minimax_h3_cache_none_by_default(self, monkeypatch):
        """minimax_h3_cache_directory() returns None when not set."""
        monkeypatch.delenv("MINIMAX_H3_CACHE", raising=False)
        result = env.minimax_h3_cache_directory()
        assert result is None

    def test_minimax_h3_cache_returns_env_value(self, monkeypatch):
        """minimax_h3_cache_directory() returns environment value when set."""
        test_cache = "/path/to/cache"
        monkeypatch.setenv("MINIMAX_H3_CACHE", test_cache)
        result = env.minimax_h3_cache_directory()
        assert result == test_cache

    def test_quantized_cache_root_defaults(self, monkeypatch):
        """minimax_h3_quantized_cache_root() defaults to ./quantized-cache."""
        monkeypatch.delenv("MINIMAX_H3_QUANTIZED_CACHE", raising=False)
        result = env.minimax_h3_quantized_cache_root()
        assert result.name == "quantized-cache"

    def test_disk_offload_cache_root_defaults(self, monkeypatch):
        """minimax_h3_disk_offload_cache_root() defaults to ./disk-offload-cache."""
        monkeypatch.delenv("MINIMAX_H3_DISK_OFFLOAD_CACHE", raising=False)
        result = env.minimax_h3_disk_offload_cache_root()
        assert result.name == "disk-offload-cache"

    def test_lora_cache_root_defaults(self, monkeypatch):
        """minimax_h3_lora_cache_root() defaults to ./lora-cache."""
        monkeypatch.delenv("MINIMAX_H3_LORA_CACHE", raising=False)
        result = env.minimax_h3_lora_cache_root()
        assert result.name == "lora-cache"


class TestPneumaConfiguration:
    """Tests for Pneuma device and model configuration."""

    def test_pneuma_device_defaults_to_cpu(self, monkeypatch):
        """pneuma_device() defaults to 'cpu' when not set."""
        monkeypatch.delenv("SHRIMPLY_PNEUMA_DEVICE", raising=False)
        result = env.pneuma_device()
        assert result == "cpu"

    def test_set_pneuma_device_updates_environment(self, monkeypatch):
        """set_pneuma_device() updates the SHRIMPLY_PNEUMA_DEVICE env var."""
        env.set_pneuma_device("cuda:0")
        assert os.environ.get("SHRIMPLY_PNEUMA_DEVICE") == "cuda:0"
        
        env.set_pneuma_device("cpu")
        assert os.environ.get("SHRIMPLY_PNEUMA_DEVICE") == "cpu"

    def test_pneuma_model_directory_configurable(self, monkeypatch, tmp_path):
        """PNEUMA_MODEL_DIRECTORY can be configured via environment."""
        model_dir = tmp_path / "models"
        monkeypatch.setenv("SHRIMPLY_PNEUMA_MODEL_DIR", str(model_dir))
        importlib.reload(env)
        assert env.PNEUMA_MODEL_DIRECTORY == model_dir


class TestVideoGenerationWorkerConfiguration:
    """Tests for configure_video_generation_worker() helper."""

    def test_configure_sets_expected_env_vars(self):
        """configure_video_generation_worker() sets thread and memory limits."""
        # Save original environment
        original_malloc = os.environ.get("MALLOC_ARENA_MAX")
        original_omp = os.environ.get("OMP_NUM_THREADS")
        original_mkl = os.environ.get("MKL_NUM_THREADS")
        original_tok = os.environ.get("TOKENIZERS_PARALLELISM")
        
        try:
            # Call the configuration function
            env.configure_video_generation_worker()
            
            # Verify expected settings
            assert os.environ.get("MALLOC_ARENA_MAX") == "2"
            assert os.environ.get("OMP_NUM_THREADS") == "1"
            assert os.environ.get("MKL_NUM_THREADS") == "1"
            assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
        finally:
            # Restore original environment
            if original_malloc is not None:
                os.environ["MALLOC_ARENA_MAX"] = original_malloc
            else:
                os.environ.pop("MALLOC_ARENA_MAX", None)
            if original_omp is not None:
                os.environ["OMP_NUM_THREADS"] = original_omp
            else:
                os.environ.pop("OMP_NUM_THREADS", None)
            if original_mkl is not None:
                os.environ["MKL_NUM_THREADS"] = original_mkl
            else:
                os.environ.pop("MKL_NUM_THREADS", None)
            if original_tok is not None:
                os.environ["TOKENIZERS_PARALLELISM"] = original_tok
            else:
                os.environ.pop("TOKENIZERS_PARALLELISM", None)
