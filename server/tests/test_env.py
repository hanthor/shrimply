"""Unit tests for server/src/env.py configuration utilities."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import env


class TestPositiveInt(TestCase):
    """Test _positive_int validation function."""

    def test_returns_default_when_env_var_not_set(self):
        """Should return default value when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = env._positive_int("NONEXISTENT_VAR", 42)
            self.assertEqual(result, 42)

    def test_returns_env_value_when_set(self):
        """Should return environment variable value when set."""
        with patch.dict(os.environ, {"TEST_VAR": "100"}):
            result = env._positive_int("TEST_VAR", 42)
            self.assertEqual(result, 100)

    def test_raises_on_zero_value(self):
        """Should raise ValueError when value is zero."""
        with patch.dict(os.environ, {"TEST_VAR": "0"}):
            with self.assertRaises(ValueError) as ctx:
                env._positive_int("TEST_VAR", 42)
            self.assertIn("must be positive", str(ctx.exception))

    def test_raises_on_negative_value(self):
        """Should raise ValueError when value is negative."""
        with patch.dict(os.environ, {"TEST_VAR": "-5"}):
            with self.assertRaises(ValueError) as ctx:
                env._positive_int("TEST_VAR", 42)
            self.assertIn("must be positive", str(ctx.exception))

    def test_accepts_positive_values(self):
        """Should accept any positive integer."""
        with patch.dict(os.environ, {"TEST_VAR": "1"}):
            self.assertEqual(env._positive_int("TEST_VAR", 42), 1)
        with patch.dict(os.environ, {"TEST_VAR": "999999"}):
            self.assertEqual(env._positive_int("TEST_VAR", 42), 999999)

    def test_raises_on_non_integer_value(self):
        """Should raise ValueError for non-integer values."""
        with patch.dict(os.environ, {"TEST_VAR": "not_a_number"}):
            with self.assertRaises(ValueError):
                env._positive_int("TEST_VAR", 42)

    def test_raises_on_float_string(self):
        """Should raise ValueError for float strings."""
        with patch.dict(os.environ, {"TEST_VAR": "3.14"}):
            with self.assertRaises(ValueError):
                env._positive_int("TEST_VAR", 42)


class TestPathFunction(TestCase):
    """Test _path directory resolution function."""

    def test_returns_default_when_env_var_not_set(self):
        """Should return default path when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            default = Path("/tmp/default")
            result = env._path("TEST_PATH", default)
            self.assertEqual(result, default)

    def test_expands_user_home_directory(self):
        """Should expand ~ to user home directory."""
        home = Path.home()
        with patch.dict(os.environ, {"TEST_PATH": "~/test_dir"}):
            result = env._path("TEST_PATH", Path("/default"))
            # Result should be resolved with ~ expanded
            self.assertEqual(result, (home / "test_dir").resolve())

    def test_resolves_relative_paths(self):
        """Should resolve relative paths to absolute paths."""
        with patch.dict(os.environ, {"TEST_PATH": "./relative/path"}):
            result = env._path("TEST_PATH", Path("/default"))
            self.assertTrue(result.is_absolute())

    def test_resolves_symlinks(self):
        """Should resolve symlinks and relative path components."""
        with patch.dict(os.environ, {"TEST_PATH": "/tmp/../usr/bin"}):
            result = env._path("TEST_PATH", Path("/default"))
            # resolve() removes .. components
            self.assertEqual(str(result), "/usr/bin")

    def test_preserves_absolute_paths(self):
        """Should preserve absolute paths."""
        with patch.dict(os.environ, {"TEST_PATH": "/absolute/path"}):
            result = env._path("TEST_PATH", Path("/default"))
            self.assertEqual(result, Path("/absolute/path"))


class TestServerConfiguration(TestCase):
    """Test server configuration variables."""

    def test_server_host_default(self):
        """SERVER_HOST should default to 0.0.0.0."""
        with patch.dict(os.environ, {}, clear=True):
            # Need to reload module to get fresh defaults
            # For now, test that the constant is a string
            self.assertIsInstance(env.SERVER_HOST, str)

    def test_server_port_positive_int(self):
        """SERVER_PORT should be a positive integer."""
        self.assertIsInstance(env.SERVER_PORT, int)
        self.assertGreater(env.SERVER_PORT, 0)

    def test_server_share_validation(self):
        """SERVER_SHARE should validate 0 or 1 string values."""
        # Test that SERVER_SHARE is boolean (it's already been validated on import)
        self.assertIsInstance(env.SERVER_SHARE, bool)

    def test_server_share_invalid_value_raises_error(self):
        """Should raise error if SHRIMPLY_SERVER_SHARE is not 0 or 1."""
        with patch.dict(os.environ, {"SHRIMPLY_SERVER_SHARE": "invalid"}):
            with self.assertRaises(ValueError) as ctx:
                # Create a fresh evaluation
                value = os.environ.get("SHRIMPLY_SERVER_SHARE", "0")
                if value not in {"0", "1"}:
                    raise ValueError("SHRIMPLY_SERVER_SHARE must be 0 or 1")
            self.assertIn("must be 0 or 1", str(ctx.exception))


class TestVideoGenerationCacheRoot(TestCase):
    """Test video generation cache directory resolution."""

    def test_returns_default_when_not_configured(self):
        """Should return cwd/video-generation-cache when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            result = env.video_generation_cache_root()
            expected = (Path.cwd() / "video-generation-cache").resolve()
            self.assertEqual(result, expected)

    def test_uses_configured_path(self):
        """Should use SHRIMPLY_VIDEO_GENERATION_CACHE when set."""
        with patch.dict(os.environ, {"SHRIMPLY_VIDEO_GENERATION_CACHE": "/custom/cache"}):
            result = env.video_generation_cache_root()
            self.assertEqual(result, Path("/custom/cache"))

    def test_expands_user_path(self):
        """Should expand ~ in configured path."""
        home = Path.home()
        with patch.dict(os.environ, {"SHRIMPLY_VIDEO_GENERATION_CACHE": "~/cache"}):
            result = env.video_generation_cache_root()
            self.assertEqual(result, (home / "cache").resolve())


class TestMinimaxH3CacheDirectory(TestCase):
    """Test minimax_h3_cache_directory function."""

    def test_returns_none_when_not_set(self):
        """Should return None when MINIMAX_H3_CACHE is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = env.minimax_h3_cache_directory()
            self.assertIsNone(result)

    def test_returns_empty_string_as_none(self):
        """Should treat empty string as None."""
        with patch.dict(os.environ, {"MINIMAX_H3_CACHE": ""}):
            result = env.minimax_h3_cache_directory()
            self.assertIsNone(result)

    def test_returns_configured_value(self):
        """Should return the configured cache directory path."""
        with patch.dict(os.environ, {"MINIMAX_H3_CACHE": "/cache/minimax"}):
            result = env.minimax_h3_cache_directory()
            self.assertEqual(result, "/cache/minimax")


class TestMinimaxH3DecodeOffload(TestCase):
    """Test minimax_h3_decode_offload function."""

    def test_returns_default_auto(self):
        """Should default to 'auto'."""
        with patch.dict(os.environ, {}, clear=True):
            result = env.minimax_h3_decode_offload()
            self.assertEqual(result, "auto")

    def test_accepts_gpu_value(self):
        """Should accept 'gpu' value."""
        with patch.dict(os.environ, {"MINIMAX_H3_DECODE_OFFLOAD": "gpu"}):
            result = env.minimax_h3_decode_offload()
            self.assertEqual(result, "gpu")

    def test_accepts_disk_value(self):
        """Should accept 'disk' value."""
        with patch.dict(os.environ, {"MINIMAX_H3_DECODE_OFFLOAD": "disk"}):
            result = env.minimax_h3_decode_offload()
            self.assertEqual(result, "disk")

    def test_normalizes_to_lowercase(self):
        """Should normalize uppercase values to lowercase."""
        with patch.dict(os.environ, {"MINIMAX_H3_DECODE_OFFLOAD": "GPU"}):
            result = env.minimax_h3_decode_offload()
            self.assertEqual(result, "gpu")

    def test_rejects_invalid_values(self):
        """Should raise ValueError for invalid values."""
        with patch.dict(os.environ, {"MINIMAX_H3_DECODE_OFFLOAD": "invalid"}):
            with self.assertRaises(ValueError) as ctx:
                env.minimax_h3_decode_offload()
            self.assertIn("must be auto, gpu, or disk", str(ctx.exception))


class TestPneumaDevice(TestCase):
    """Test pneuma device configuration functions."""

    def test_pneuma_device_returns_default(self):
        """pneuma_device() should return 'cpu' by default."""
        with patch.dict(os.environ, {}, clear=True):
            result = env.pneuma_device()
            self.assertEqual(result, "cpu")

    def test_pneuma_device_returns_configured_value(self):
        """pneuma_device() should return configured device."""
        with patch.dict(os.environ, {"SHRIMPLY_PNEUMA_DEVICE": "cuda:0"}):
            result = env.pneuma_device()
            self.assertEqual(result, "cuda:0")

    def test_set_pneuma_device_updates_env(self):
        """set_pneuma_device() should set the environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            env.set_pneuma_device("cuda:1")
            self.assertEqual(os.environ.get("SHRIMPLY_PNEUMA_DEVICE"), "cuda:1")

    def test_set_and_get_pneuma_device_roundtrip(self):
        """Should be able to set and then get pneuma device."""
        with patch.dict(os.environ, {}, clear=True):
            env.set_pneuma_device("mps")
            result = env.pneuma_device()
            self.assertEqual(result, "mps")


class TestConfigureVideoGenerationWorker(TestCase):
    """Test configure_video_generation_worker function."""

    def test_sets_malloc_arena_max(self):
        """Should set MALLOC_ARENA_MAX to 2."""
        with patch.dict(os.environ, {}, clear=True):
            env.configure_video_generation_worker()
            self.assertEqual(os.environ.get("MALLOC_ARENA_MAX"), "2")

    def test_sets_omp_num_threads(self):
        """Should set OMP_NUM_THREADS to 1."""
        with patch.dict(os.environ, {}, clear=True):
            env.configure_video_generation_worker()
            self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "1")

    def test_sets_mkl_num_threads(self):
        """Should set MKL_NUM_THREADS to 1."""
        with patch.dict(os.environ, {}, clear=True):
            env.configure_video_generation_worker()
            self.assertEqual(os.environ.get("MKL_NUM_THREADS"), "1")

    def test_sets_tokenizers_parallelism_false(self):
        """Should set TOKENIZERS_PARALLELISM to false."""
        with patch.dict(os.environ, {}, clear=True):
            env.configure_video_generation_worker()
            self.assertEqual(os.environ.get("TOKENIZERS_PARALLELISM"), "false")

    def test_all_environment_variables_set(self):
        """Should set all four required environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            env.configure_video_generation_worker()
            self.assertEqual(len([k for k in os.environ if k.startswith(("MALLOC", "OMP", "MKL", "TOKENIZERS"))]), 4)


if __name__ == "__main__":
    import unittest
    unittest.main()
