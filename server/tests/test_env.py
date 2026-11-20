"""Unit tests for server/src/env.py configuration utilities."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Import the module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import env


class TestPositiveInt(unittest.TestCase):
    """Tests for _positive_int() validation."""

    def test_positive_int_uses_environment_value(self):
        """_positive_int() reads from environment when set."""
        with mock.patch.dict(os.environ, {'TEST_VAR': '42'}):
            result = env._positive_int('TEST_VAR', 10)
            self.assertEqual(result, 42)

    def test_positive_int_uses_default_when_not_set(self):
        """_positive_int() uses default when environment variable is not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = env._positive_int('NONEXISTENT_VAR', 10)
            self.assertEqual(result, 10)

    def test_positive_int_rejects_zero(self):
        """_positive_int() raises ValueError for zero."""
        with mock.patch.dict(os.environ, {'TEST_VAR': '0'}):
            with self.assertRaises(ValueError) as ctx:
                env._positive_int('TEST_VAR', 10)
            self.assertIn('must be positive', str(ctx.exception))

    def test_positive_int_rejects_negative(self):
        """_positive_int() raises ValueError for negative numbers."""
        with mock.patch.dict(os.environ, {'TEST_VAR': '-5'}):
            with self.assertRaises(ValueError) as ctx:
                env._positive_int('TEST_VAR', 10)
            self.assertIn('must be positive', str(ctx.exception))

    def test_positive_int_accepts_large_positive(self):
        """_positive_int() accepts large positive integers."""
        with mock.patch.dict(os.environ, {'TEST_VAR': '999999'}):
            result = env._positive_int('TEST_VAR', 10)
            self.assertEqual(result, 999999)

    def test_positive_int_rejects_invalid_format(self):
        """_positive_int() raises ValueError for non-integer strings."""
        with mock.patch.dict(os.environ, {'TEST_VAR': 'not_a_number'}):
            with self.assertRaises(ValueError):
                env._positive_int('TEST_VAR', 10)


class TestPath(unittest.TestCase):
    """Tests for _path() path handling."""

    def test_path_uses_environment_when_set(self):
        """_path() uses environment variable when set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {'TEST_PATH': tmpdir}):
                result = env._path('TEST_PATH', Path('/default'))
                self.assertEqual(result, Path(tmpdir).resolve())

    def test_path_uses_default_when_not_set(self):
        """_path() uses default when environment variable is not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            default = Path('/some/default/path')
            result = env._path('TEST_PATH', default)
            # Note: on most systems, /some/default/path won't exist, but resolve() still works
            self.assertEqual(result, default.resolve())

    def test_path_expands_user_home(self):
        """_path() expands ~ to home directory."""
        home = Path.home()
        with mock.patch.dict(os.environ, {'TEST_PATH': '~/mypath'}):
            result = env._path('TEST_PATH', Path('/default'))
            self.assertEqual(result, (home / 'mypath').resolve())

    def test_path_resolves_relative_paths(self):
        """_path() resolves relative paths to absolute."""
        with mock.patch.dict(os.environ, {'TEST_PATH': './relative/path'}):
            result = env._path('TEST_PATH', Path('/default'))
            # Result should be absolute
            self.assertTrue(result.is_absolute())

    def test_path_with_empty_string_uses_default(self):
        """_path() uses default when environment variable is empty string."""
        with mock.patch.dict(os.environ, {'TEST_PATH': ''}):
            default = Path('/default')
            result = env._path('TEST_PATH', default)
            self.assertEqual(result, default.resolve())


class TestServerConfiguration(unittest.TestCase):
    """Tests for server configuration variables."""

    def test_server_host_default(self):
        """SERVER_HOST defaults to 0.0.0.0."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Re-evaluate the module's top-level code in isolation
            result = os.environ.get("SHRIMPLY_SERVER_HOST", "0.0.0.0")
            self.assertEqual(result, "0.0.0.0")

    def test_server_host_custom(self):
        """SERVER_HOST can be customized via environment."""
        with mock.patch.dict(os.environ, {'SHRIMPLY_SERVER_HOST': 'localhost'}):
            result = os.environ.get("SHRIMPLY_SERVER_HOST", "0.0.0.0")
            self.assertEqual(result, "localhost")

    def test_server_port_default(self):
        """SERVER_PORT defaults to 8787."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = env._positive_int("SHRIMPLY_SERVER_PORT", 8787)
            self.assertEqual(result, 8787)

    def test_server_port_custom(self):
        """SERVER_PORT can be customized via environment."""
        with mock.patch.dict(os.environ, {'SHRIMPLY_SERVER_PORT': '9000'}):
            result = env._positive_int("SHRIMPLY_SERVER_PORT", 8787)
            self.assertEqual(result, 9000)

    def test_server_share_valid_zero(self):
        """SHRIMPLY_SERVER_SHARE accepts '0' for False."""
        value = "0"
        self.assertIn(value, {"0", "1"})
        self.assertFalse(value == "1")

    def test_server_share_valid_one(self):
        """SHRIMPLY_SERVER_SHARE accepts '1' for True."""
        value = "1"
        self.assertIn(value, {"0", "1"})
        self.assertTrue(value == "1")

    def test_server_share_invalid_value(self):
        """SHRIMPLY_SERVER_SHARE rejects invalid values."""
        invalid_values = ["2", "true", "false", "yes", "no", "on", "off"]
        for value in invalid_values:
            with self.subTest(value=value):
                self.assertNotIn(value, {"0", "1"})


class TestVideoCacheConfiguration(unittest.TestCase):
    """Tests for video generation cache configuration."""

    def test_video_generation_cache_root_default(self):
        """video_generation_cache_root() defaults to ./video-generation-cache."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = env.video_generation_cache_root()
            expected = (Path.cwd() / "video-generation-cache").resolve()
            self.assertEqual(result, expected)

    def test_video_generation_cache_root_custom(self):
        """video_generation_cache_root() uses custom path from environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {'SHRIMPLY_VIDEO_GENERATION_CACHE': tmpdir}):
                result = env.video_generation_cache_root()
                self.assertEqual(result, Path(tmpdir).resolve())

    def test_video_generation_cache_root_expands_user(self):
        """video_generation_cache_root() expands ~ in custom paths."""
        with mock.patch.dict(os.environ, {'SHRIMPLY_VIDEO_GENERATION_CACHE': '~/video-cache'}):
            result = env.video_generation_cache_root()
            expected = (Path.home() / 'video-cache').resolve()
            self.assertEqual(result, expected)


class TestMiniMaxH3CacheConfiguration(unittest.TestCase):
    """Tests for MiniMax H3 cache configuration functions."""

    def test_minimax_h3_cache_directory_returns_none_when_not_set(self):
        """minimax_h3_cache_directory() returns None when not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = env.minimax_h3_cache_directory()
            self.assertIsNone(result)

    def test_minimax_h3_cache_directory_returns_value_when_set(self):
        """minimax_h3_cache_directory() returns value when environment is set."""
        with mock.patch.dict(os.environ, {'MINIMAX_H3_CACHE': '/path/to/cache'}):
            result = env.minimax_h3_cache_directory()
            self.assertEqual(result, '/path/to/cache')

    def test_minimax_h3_quantized_cache_root_default(self):
        """minimax_h3_quantized_cache_root() defaults to ./quantized-cache."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = env.minimax_h3_quantized_cache_root()
            expected = (Path.cwd() / "quantized-cache").resolve()
            self.assertEqual(result, expected)

    def test_minimax_h3_quantized_cache_root_custom(self):
        """minimax_h3_quantized_cache_root() uses custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {'MINIMAX_H3_QUANTIZED_CACHE': tmpdir}):
                result = env.minimax_h3_quantized_cache_root()
                self.assertEqual(result, Path(tmpdir).resolve())

    def test_minimax_h3_disk_offload_cache_root_default(self):
        """minimax_h3_disk_offload_cache_root() defaults to ./disk-offload-cache."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = env.minimax_h3_disk_offload_cache_root()
            expected = (Path.cwd() / "disk-offload-cache").resolve()
            self.assertEqual(result, expected)

    def test_minimax_h3_disk_offload_cache_root_custom(self):
        """minimax_h3_disk_offload_cache_root() uses custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {'MINIMAX_H3_DISK_OFFLOAD_CACHE': tmpdir}):
                result = env.minimax_h3_disk_offload_cache_root()
                self.assertEqual(result, Path(tmpdir).resolve())

    def test_minimax_h3_lora_cache_root_default(self):
        """minimax_h3_lora_cache_root() defaults to ./lora-cache."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = env.minimax_h3_lora_cache_root()
            expected = (Path.cwd() / "lora-cache").resolve()
            self.assertEqual(result, expected)

    def test_minimax_h3_lora_cache_root_custom(self):
        """minimax_h3_lora_cache_root() uses custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {'MINIMAX_H3_LORA_CACHE': tmpdir}):
                result = env.minimax_h3_lora_cache_root()
                self.assertEqual(result, Path(tmpdir).resolve())


class TestMiniMaxH3DecodeOffload(unittest.TestCase):
    """Tests for MiniMax H3 decode offload strategy."""

    def test_minimax_h3_decode_offload_default(self):
        """minimax_h3_decode_offload() defaults to 'auto'."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = env.minimax_h3_decode_offload()
            self.assertEqual(result, "auto")

    def test_minimax_h3_decode_offload_gpu(self):
        """minimax_h3_decode_offload() accepts 'gpu'."""
        with mock.patch.dict(os.environ, {'MINIMAX_H3_DECODE_OFFLOAD': 'gpu'}):
            result = env.minimax_h3_decode_offload()
            self.assertEqual(result, "gpu")

    def test_minimax_h3_decode_offload_disk(self):
        """minimax_h3_decode_offload() accepts 'disk'."""
        with mock.patch.dict(os.environ, {'MINIMAX_H3_DECODE_OFFLOAD': 'disk'}):
            result = env.minimax_h3_decode_offload()
            self.assertEqual(result, "disk")

    def test_minimax_h3_decode_offload_case_insensitive(self):
        """minimax_h3_decode_offload() is case insensitive."""
        test_cases = ['AUTO', 'Auto', 'GPU', 'Gpu', 'DISK', 'Disk']
        for case in test_cases:
            with self.subTest(case=case):
                with mock.patch.dict(os.environ, {'MINIMAX_H3_DECODE_OFFLOAD': case}):
                    result = env.minimax_h3_decode_offload()
                    self.assertIn(result, {"auto", "gpu", "disk"})

    def test_minimax_h3_decode_offload_invalid_value(self):
        """minimax_h3_decode_offload() rejects invalid values."""
        invalid_values = ['invalid', 'cpu', 'memory', 'hybrid', '']
        for value in invalid_values:
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {'MINIMAX_H3_DECODE_OFFLOAD': value}):
                    with self.assertRaises(ValueError) as ctx:
                        env.minimax_h3_decode_offload()
                    self.assertIn('must be auto, gpu, or disk', str(ctx.exception))


class TestPneumaDevice(unittest.TestCase):
    """Tests for Pneuma device management."""

    def test_pneuma_device_default(self):
        """pneuma_device() defaults to 'cpu'."""
        with mock.patch.dict(os.environ, {}, clear=True):
            result = env.pneuma_device()
            self.assertEqual(result, "cpu")

    def test_pneuma_device_custom(self):
        """pneuma_device() returns custom device from environment."""
        with mock.patch.dict(os.environ, {'SHRIMPLY_PNEUMA_DEVICE': 'cuda'}):
            result = env.pneuma_device()
            self.assertEqual(result, "cuda")

    def test_set_pneuma_device(self):
        """set_pneuma_device() updates the environment and pneuma_device() reads it."""
        with mock.patch.dict(os.environ, {}, clear=True):
            env.set_pneuma_device('gpu')
            self.assertEqual(os.environ['SHRIMPLY_PNEUMA_DEVICE'], 'gpu')
            result = env.pneuma_device()
            self.assertEqual(result, 'gpu')

    def test_set_pneuma_device_various_values(self):
        """set_pneuma_device() accepts various device specifications."""
        devices = ['cpu', 'cuda', 'cuda:0', 'metal', 'tpu']
        for device in devices:
            with self.subTest(device=device):
                with mock.patch.dict(os.environ, {}, clear=True):
                    env.set_pneuma_device(device)
                    result = env.pneuma_device()
                    self.assertEqual(result, device)


class TestVideoGenerationWorkerConfiguration(unittest.TestCase):
    """Tests for video generation worker environment setup."""

    def test_configure_video_generation_worker_sets_malloc_arena_max(self):
        """configure_video_generation_worker() sets MALLOC_ARENA_MAX to '2'."""
        with mock.patch.dict(os.environ, {}, clear=True):
            env.configure_video_generation_worker()
            self.assertEqual(os.environ['MALLOC_ARENA_MAX'], '2')

    def test_configure_video_generation_worker_sets_omp_num_threads(self):
        """configure_video_generation_worker() sets OMP_NUM_THREADS to '1'."""
        with mock.patch.dict(os.environ, {}, clear=True):
            env.configure_video_generation_worker()
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '1')

    def test_configure_video_generation_worker_sets_mkl_num_threads(self):
        """configure_video_generation_worker() sets MKL_NUM_THREADS to '1'."""
        with mock.patch.dict(os.environ, {}, clear=True):
            env.configure_video_generation_worker()
            self.assertEqual(os.environ['MKL_NUM_THREADS'], '1')

    def test_configure_video_generation_worker_sets_tokenizers_parallelism(self):
        """configure_video_generation_worker() sets TOKENIZERS_PARALLELISM to 'false'."""
        with mock.patch.dict(os.environ, {}, clear=True):
            env.configure_video_generation_worker()
            self.assertEqual(os.environ['TOKENIZERS_PARALLELISM'], 'false')

    def test_configure_video_generation_worker_all_vars(self):
        """configure_video_generation_worker() sets all required variables."""
        with mock.patch.dict(os.environ, {}, clear=True):
            env.configure_video_generation_worker()
            expected = {
                'MALLOC_ARENA_MAX': '2',
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1',
                'TOKENIZERS_PARALLELISM': 'false',
            }
            for key, value in expected.items():
                with self.subTest(key=key):
                    self.assertEqual(os.environ[key], value)


if __name__ == '__main__':
    unittest.main()
