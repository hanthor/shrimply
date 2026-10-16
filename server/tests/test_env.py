"""Unit tests for server/src/env.py configuration utilities."""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch


class TestPositiveInt(TestCase):
    """Tests for _positive_int() validation."""

    def setUp(self):
        """Save and clear relevant environment variables."""
        self.saved_env = {}
        for name in ["TEST_POSITIVE_INT", "SHRIMPLY_SERVER_PORT"]:
            self.saved_env[name] = os.environ.get(name)
            if name in os.environ:
                del os.environ[name]

    def tearDown(self):
        """Restore environment variables."""
        for name, value in self.saved_env.items():
            if value is not None:
                os.environ[name] = value
            elif name in os.environ:
                del os.environ[name]

    def test_positive_int_returns_env_value_when_set(self):
        """Returns environment value when variable is set."""
        os.environ["TEST_POSITIVE_INT"] = "42"
        # Import after setting env to get fresh module
        from server.src import env
        result = env._positive_int("TEST_POSITIVE_INT", 10)
        self.assertEqual(result, 42)

    def test_positive_int_returns_default_when_not_set(self):
        """Returns default value when environment variable is not set."""
        from server.src import env
        result = env._positive_int("TEST_POSITIVE_INT", 100)
        self.assertEqual(result, 100)

    def test_positive_int_rejects_zero(self):
        """Raises ValueError when value is zero."""
        os.environ["TEST_POSITIVE_INT"] = "0"
        from server.src import env
        with self.assertRaises(ValueError) as ctx:
            env._positive_int("TEST_POSITIVE_INT", 10)
        self.assertIn("must be positive", str(ctx.exception))

    def test_positive_int_rejects_negative(self):
        """Raises ValueError when value is negative."""
        os.environ["TEST_POSITIVE_INT"] = "-5"
        from server.src import env
        with self.assertRaises(ValueError) as ctx:
            env._positive_int("TEST_POSITIVE_INT", 10)
        self.assertIn("must be positive", str(ctx.exception))

    def test_positive_int_rejects_non_numeric_string(self):
        """Raises ValueError when environment value is not numeric."""
        os.environ["TEST_POSITIVE_INT"] = "not_a_number"
        from server.src import env
        with self.assertRaises(ValueError):
            env._positive_int("TEST_POSITIVE_INT", 10)

    def test_positive_int_with_large_value(self):
        """Accepts large positive integers."""
        os.environ["TEST_POSITIVE_INT"] = "999999999"
        from server.src import env
        result = env._positive_int("TEST_POSITIVE_INT", 10)
        self.assertEqual(result, 999999999)


class TestPathFunction(TestCase):
    """Tests for _path() path expansion and resolution."""

    def setUp(self):
        """Clear environment variables."""
        self.saved_env = {}
        for name in ["TEST_PATH", "HOME"]:
            self.saved_env[name] = os.environ.get(name)

    def tearDown(self):
        """Restore environment variables."""
        for name, value in self.saved_env.items():
            if value is not None:
                os.environ[name] = value
            elif name in os.environ:
                del os.environ[name]

    def test_path_returns_default_when_not_set(self):
        """Returns default path when environment variable is not set."""
        if "TEST_PATH" in os.environ:
            del os.environ["TEST_PATH"]
        from server.src import env
        default = Path("/default/path")
        result = env._path("TEST_PATH", default)
        self.assertEqual(result, default)

    def test_path_expands_home_directory(self):
        """Expands ~ to user home directory."""
        with TemporaryDirectory() as tmpdir:
            test_home = Path(tmpdir)
            os.environ["HOME"] = str(test_home)
            os.environ["TEST_PATH"] = "~/subdir"
            from server.src import env
            # Force reload to use new HOME
            import importlib
            importlib.reload(env)
            result = env._path("TEST_PATH", Path("/default"))
            # Result should be resolved (absolute) path
            self.assertTrue(result.is_absolute())
            self.assertEqual(result.name, "subdir")

    def test_path_resolves_to_absolute(self):
        """Resolves relative paths to absolute."""
        os.environ["TEST_PATH"] = "."
        from server.src import env
        result = env._path("TEST_PATH", Path("/default"))
        self.assertTrue(result.is_absolute())

    def test_path_with_absolute_path(self):
        """Accepts absolute paths as-is (resolved)."""
        os.environ["TEST_PATH"] = "/absolute/path"
        from server.src import env
        result = env._path("TEST_PATH", Path("/default"))
        self.assertEqual(result, Path("/absolute/path"))

    def test_path_with_relative_components(self):
        """Handles relative path components (.. .)."""
        with TemporaryDirectory() as tmpdir:
            os.environ["TEST_PATH"] = f"{tmpdir}/sub/../file"
            from server.src import env
            result = env._path("TEST_PATH", Path("/default"))
            # Should resolve .. correctly
            self.assertTrue(result.is_absolute())
            self.assertNotIn("..", str(result))


class TestServerHost(TestCase):
    """Tests for SERVER_HOST configuration."""

    def test_server_host_default_value(self):
        """SERVER_HOST defaults to 0.0.0.0."""
        if "SHRIMPLY_SERVER_HOST" in os.environ:
            del os.environ["SHRIMPLY_SERVER_HOST"]
        # Reload to pick up env change
        import importlib
        from server.src import env
        importlib.reload(env)
        self.assertEqual(env.SERVER_HOST, "0.0.0.0")

    def test_server_host_custom_value(self):
        """SERVER_HOST can be set via environment variable."""
        os.environ["SHRIMPLY_SERVER_HOST"] = "127.0.0.1"
        import importlib
        from server.src import env
        importlib.reload(env)
        self.assertEqual(env.SERVER_HOST, "127.0.0.1")


class TestServerPort(TestCase):
    """Tests for SERVER_PORT configuration."""

    def setUp(self):
        """Clear SERVER_PORT from environment."""
        if "SHRIMPLY_SERVER_PORT" in os.environ:
            del os.environ["SHRIMPLY_SERVER_PORT"]

    def tearDown(self):
        """Clean up."""
        if "SHRIMPLY_SERVER_PORT" in os.environ:
            del os.environ["SHRIMPLY_SERVER_PORT"]

    def test_server_port_default_value(self):
        """SERVER_PORT defaults to 8787."""
        import importlib
        from server.src import env
        importlib.reload(env)
        self.assertEqual(env.SERVER_PORT, 8787)

    def test_server_port_custom_value(self):
        """SERVER_PORT can be set via environment variable."""
        os.environ["SHRIMPLY_SERVER_PORT"] = "9000"
        import importlib
        from server.src import env
        importlib.reload(env)
        self.assertEqual(env.SERVER_PORT, 9000)

    def test_server_port_rejects_zero(self):
        """SERVER_PORT rejects port 0."""
        os.environ["SHRIMPLY_SERVER_PORT"] = "0"
        import importlib
        from server.src import env
        with self.assertRaises(ValueError):
            importlib.reload(env)

    def test_server_port_rejects_negative(self):
        """SERVER_PORT rejects negative values."""
        os.environ["SHRIMPLY_SERVER_PORT"] = "-1"
        import importlib
        from server.src import env
        with self.assertRaises(ValueError):
            importlib.reload(env)


class TestServerShare(TestCase):
    """Tests for SERVER_SHARE boolean configuration."""

    def setUp(self):
        """Clear SERVER_SHARE from environment."""
        if "SHRIMPLY_SERVER_SHARE" in os.environ:
            del os.environ["SHRIMPLY_SERVER_SHARE"]

    def tearDown(self):
        """Clean up."""
        if "SHRIMPLY_SERVER_SHARE" in os.environ:
            del os.environ["SHRIMPLY_SERVER_SHARE"]

    def test_server_share_default_false(self):
        """SERVER_SHARE defaults to False (0)."""
        import importlib
        from server.src import env
        importlib.reload(env)
        self.assertFalse(env.SERVER_SHARE)

    def test_server_share_true_value(self):
        """SERVER_SHARE is True when set to 1."""
        os.environ["SHRIMPLY_SERVER_SHARE"] = "1"
        import importlib
        from server.src import env
        importlib.reload(env)
        self.assertTrue(env.SERVER_SHARE)

    def test_server_share_false_value(self):
        """SERVER_SHARE is False when set to 0."""
        os.environ["SHRIMPLY_SERVER_SHARE"] = "0"
        import importlib
        from server.src import env
        importlib.reload(env)
        self.assertFalse(env.SERVER_SHARE)

    def test_server_share_rejects_invalid_values(self):
        """SERVER_SHARE rejects values other than 0 or 1."""
        os.environ["SHRIMPLY_SERVER_SHARE"] = "2"
        import importlib
        from server.src import env
        with self.assertRaises(ValueError) as ctx:
            importlib.reload(env)
        self.assertIn("must be 0 or 1", str(ctx.exception))

    def test_server_share_rejects_string_values(self):
        """SERVER_SHARE rejects non-numeric strings."""
        os.environ["SHRIMPLY_SERVER_SHARE"] = "true"
        import importlib
        from server.src import env
        with self.assertRaises(ValueError):
            importlib.reload(env)


class TestVideoGenerationCacheRoot(TestCase):
    """Tests for video_generation_cache_root() function."""

    def setUp(self):
        """Clear cache root environment variable."""
        if "SHRIMPLY_VIDEO_GENERATION_CACHE" in os.environ:
            del os.environ["SHRIMPLY_VIDEO_GENERATION_CACHE"]

    def tearDown(self):
        """Clean up."""
        if "SHRIMPLY_VIDEO_GENERATION_CACHE" in os.environ:
            del os.environ["SHRIMPLY_VIDEO_GENERATION_CACHE"]

    def test_video_generation_cache_default_path(self):
        """Defaults to video-generation-cache in current directory."""
        from server.src import env
        result = env.video_generation_cache_root()
        self.assertTrue(result.is_absolute())
        self.assertEqual(result.name, "video-generation-cache")

    def test_video_generation_cache_custom_path(self):
        """Uses custom path from environment variable."""
        os.environ["SHRIMPLY_VIDEO_GENERATION_CACHE"] = "/custom/cache"
        from server.src import env
        result = env.video_generation_cache_root()
        self.assertEqual(result, Path("/custom/cache"))

    def test_video_generation_cache_expands_home(self):
        """Expands ~ in custom path."""
        os.environ["SHRIMPLY_VIDEO_GENERATION_CACHE"] = "~/cache"
        from server.src import env
        result = env.video_generation_cache_root()
        self.assertTrue(result.is_absolute())
        self.assertFalse("~" in str(result))


class TestMinimaxH3CacheDirectory(TestCase):
    """Tests for minimax_h3_cache_directory() function."""

    def setUp(self):
        """Clear cache directory environment variable."""
        if "MINIMAX_H3_CACHE" in os.environ:
            del os.environ["MINIMAX_H3_CACHE"]

    def tearDown(self):
        """Clean up."""
        if "MINIMAX_H3_CACHE" in os.environ:
            del os.environ["MINIMAX_H3_CACHE"]

    def test_minimax_h3_cache_directory_none_when_not_set(self):
        """Returns None when MINIMAX_H3_CACHE is not set."""
        from server.src import env
        result = env.minimax_h3_cache_directory()
        self.assertIsNone(result)

    def test_minimax_h3_cache_directory_returns_value_when_set(self):
        """Returns environment value when set."""
        os.environ["MINIMAX_H3_CACHE"] = "/cache/path"
        from server.src import env
        result = env.minimax_h3_cache_directory()
        self.assertEqual(result, "/cache/path")

    def test_minimax_h3_cache_directory_empty_string_is_none(self):
        """Treats empty string as None."""
        os.environ["MINIMAX_H3_CACHE"] = ""
        from server.src import env
        result = env.minimax_h3_cache_directory()
        self.assertIsNone(result)


class TestMinimaxH3QuantizedCacheRoot(TestCase):
    """Tests for minimax_h3_quantized_cache_root() function."""

    def setUp(self):
        """Clear cache root environment variable."""
        if "MINIMAX_H3_QUANTIZED_CACHE" in os.environ:
            del os.environ["MINIMAX_H3_QUANTIZED_CACHE"]

    def tearDown(self):
        """Clean up."""
        if "MINIMAX_H3_QUANTIZED_CACHE" in os.environ:
            del os.environ["MINIMAX_H3_QUANTIZED_CACHE"]

    def test_quantized_cache_default_path(self):
        """Defaults to quantized-cache in current directory."""
        from server.src import env
        result = env.minimax_h3_quantized_cache_root()
        self.assertTrue(result.is_absolute())
        self.assertEqual(result.name, "quantized-cache")

    def test_quantized_cache_custom_path(self):
        """Uses custom path from environment variable."""
        os.environ["MINIMAX_H3_QUANTIZED_CACHE"] = "/custom/quantized"
        from server.src import env
        result = env.minimax_h3_quantized_cache_root()
        self.assertEqual(result, Path("/custom/quantized"))


class TestMinimaxH3DiskOffloadCacheRoot(TestCase):
    """Tests for minimax_h3_disk_offload_cache_root() function."""

    def setUp(self):
        """Clear cache root environment variable."""
        if "MINIMAX_H3_DISK_OFFLOAD_CACHE" in os.environ:
            del os.environ["MINIMAX_H3_DISK_OFFLOAD_CACHE"]

    def tearDown(self):
        """Clean up."""
        if "MINIMAX_H3_DISK_OFFLOAD_CACHE" in os.environ:
            del os.environ["MINIMAX_H3_DISK_OFFLOAD_CACHE"]

    def test_disk_offload_cache_default_path(self):
        """Defaults to disk-offload-cache in current directory."""
        from server.src import env
        result = env.minimax_h3_disk_offload_cache_root()
        self.assertTrue(result.is_absolute())
        self.assertEqual(result.name, "disk-offload-cache")

    def test_disk_offload_cache_custom_path(self):
        """Uses custom path from environment variable."""
        os.environ["MINIMAX_H3_DISK_OFFLOAD_CACHE"] = "/custom/offload"
        from server.src import env
        result = env.minimax_h3_disk_offload_cache_root()
        self.assertEqual(result, Path("/custom/offload"))


class TestMinimaxH3LoraCacheRoot(TestCase):
    """Tests for minimax_h3_lora_cache_root() function."""

    def setUp(self):
        """Clear cache root environment variable."""
        if "MINIMAX_H3_LORA_CACHE" in os.environ:
            del os.environ["MINIMAX_H3_LORA_CACHE"]

    def tearDown(self):
        """Clean up."""
        if "MINIMAX_H3_LORA_CACHE" in os.environ:
            del os.environ["MINIMAX_H3_LORA_CACHE"]

    def test_lora_cache_default_path(self):
        """Defaults to lora-cache in current directory."""
        from server.src import env
        result = env.minimax_h3_lora_cache_root()
        self.assertTrue(result.is_absolute())
        self.assertEqual(result.name, "lora-cache")

    def test_lora_cache_custom_path(self):
        """Uses custom path from environment variable."""
        os.environ["MINIMAX_H3_LORA_CACHE"] = "/custom/lora"
        from server.src import env
        result = env.minimax_h3_lora_cache_root()
        self.assertEqual(result, Path("/custom/lora"))


class TestMinimaxH3DecodeOffload(TestCase):
    """Tests for minimax_h3_decode_offload() function."""

    def setUp(self):
        """Clear decode offload environment variable."""
        if "MINIMAX_H3_DECODE_OFFLOAD" in os.environ:
            del os.environ["MINIMAX_H3_DECODE_OFFLOAD"]

    def tearDown(self):
        """Clean up."""
        if "MINIMAX_H3_DECODE_OFFLOAD" in os.environ:
            del os.environ["MINIMAX_H3_DECODE_OFFLOAD"]

    def test_decode_offload_default_auto(self):
        """Defaults to 'auto' when not set."""
        from server.src import env
        result = env.minimax_h3_decode_offload()
        self.assertEqual(result, "auto")

    def test_decode_offload_gpu_value(self):
        """Accepts 'gpu' as valid value."""
        os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = "gpu"
        from server.src import env
        result = env.minimax_h3_decode_offload()
        self.assertEqual(result, "gpu")

    def test_decode_offload_disk_value(self):
        """Accepts 'disk' as valid value."""
        os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = "disk"
        from server.src import env
        result = env.minimax_h3_decode_offload()
        self.assertEqual(result, "disk")

    def test_decode_offload_case_insensitive(self):
        """Converts values to lowercase."""
        os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = "GPU"
        from server.src import env
        result = env.minimax_h3_decode_offload()
        self.assertEqual(result, "gpu")

    def test_decode_offload_rejects_invalid_value(self):
        """Raises ValueError for invalid values."""
        os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = "invalid"
        from server.src import env
        with self.assertRaises(ValueError) as ctx:
            env.minimax_h3_decode_offload()
        self.assertIn("must be auto, gpu, or disk", str(ctx.exception))


class TestPneumaDevice(TestCase):
    """Tests for pneuma_device() and set_pneuma_device() functions."""

    def setUp(self):
        """Clear device environment variable."""
        if "SHRIMPLY_PNEUMA_DEVICE" in os.environ:
            del os.environ["SHRIMPLY_PNEUMA_DEVICE"]

    def tearDown(self):
        """Clean up."""
        if "SHRIMPLY_PNEUMA_DEVICE" in os.environ:
            del os.environ["SHRIMPLY_PNEUMA_DEVICE"]

    def test_pneuma_device_default_cpu(self):
        """Defaults to 'cpu' when not set."""
        from server.src import env
        result = env.pneuma_device()
        self.assertEqual(result, "cpu")

    def test_pneuma_device_returns_set_value(self):
        """Returns value when environment variable is set."""
        os.environ["SHRIMPLY_PNEUMA_DEVICE"] = "cuda"
        from server.src import env
        result = env.pneuma_device()
        self.assertEqual(result, "cuda")

    def test_set_pneuma_device_updates_env(self):
        """set_pneuma_device() updates environment variable."""
        from server.src import env
        env.set_pneuma_device("cuda:0")
        self.assertEqual(os.environ["SHRIMPLY_PNEUMA_DEVICE"], "cuda:0")
        # Verify pneuma_device() returns the new value
        self.assertEqual(env.pneuma_device(), "cuda:0")

    def test_set_pneuma_device_overrides_existing(self):
        """set_pneuma_device() overrides existing value."""
        from server.src import env
        env.set_pneuma_device("cpu")
        self.assertEqual(env.pneuma_device(), "cpu")
        env.set_pneuma_device("cuda")
        self.assertEqual(env.pneuma_device(), "cuda")


class TestConfigureVideoGenerationWorker(TestCase):
    """Tests for configure_video_generation_worker() function."""

    def setUp(self):
        """Save environment variables for cleanup."""
        self.saved_env = {}
        for var in ["MALLOC_ARENA_MAX", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "TOKENIZERS_PARALLELISM"]:
            self.saved_env[var] = os.environ.get(var)

    def tearDown(self):
        """Restore environment variables."""
        for var, value in self.saved_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_configure_video_generation_worker_sets_all_vars(self):
        """Sets all required worker environment variables."""
        from server.src import env
        # Clear first
        for var in ["MALLOC_ARENA_MAX", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "TOKENIZERS_PARALLELISM"]:
            if var in os.environ:
                del os.environ[var]
        # Configure
        env.configure_video_generation_worker()
        # Verify all are set
        self.assertEqual(os.environ["MALLOC_ARENA_MAX"], "2")
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
        self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")
        self.assertEqual(os.environ["TOKENIZERS_PARALLELISM"], "false")

    def test_configure_video_generation_worker_overrides_existing(self):
        """Overrides existing environment variables."""
        from server.src import env
        os.environ["MALLOC_ARENA_MAX"] = "999"
        os.environ["OMP_NUM_THREADS"] = "999"
        env.configure_video_generation_worker()
        self.assertEqual(os.environ["MALLOC_ARENA_MAX"], "2")
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")


if __name__ == "__main__":
    import unittest
    unittest.main()
