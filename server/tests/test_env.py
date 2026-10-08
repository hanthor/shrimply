import os
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

from src import env


class TestPositiveInt(TestCase):
    """Tests for env._positive_int() validation."""

    def setUp(self):
        """Save original environ for restoration."""
        self.original_environ = os.environ.copy()

    def tearDown(self):
        """Restore original environ."""
        os.environ.clear()
        os.environ.update(self.original_environ)

    def test_positive_int_uses_default_when_not_set(self):
        """Return default value when environment variable is not set."""
        os.environ.pop("TEST_POSITIVE_INT", None)
        result = env._positive_int("TEST_POSITIVE_INT", 42)
        self.assertEqual(result, 42)

    def test_positive_int_accepts_positive_value(self):
        """Accept valid positive integer from environment."""
        os.environ["TEST_POSITIVE_INT"] = "100"
        result = env._positive_int("TEST_POSITIVE_INT", 42)
        self.assertEqual(result, 100)

    def test_positive_int_rejects_zero(self):
        """Reject zero as not positive."""
        os.environ["TEST_POSITIVE_INT"] = "0"
        with self.assertRaises(ValueError) as ctx:
            env._positive_int("TEST_POSITIVE_INT", 42)
        self.assertIn("must be positive", str(ctx.exception))

    def test_positive_int_rejects_negative(self):
        """Reject negative integers."""
        os.environ["TEST_POSITIVE_INT"] = "-5"
        with self.assertRaises(ValueError) as ctx:
            env._positive_int("TEST_POSITIVE_INT", 42)
        self.assertIn("must be positive", str(ctx.exception))

    def test_positive_int_rejects_non_integer(self):
        """Reject non-integer strings."""
        os.environ["TEST_POSITIVE_INT"] = "not_a_number"
        with self.assertRaises(ValueError):
            env._positive_int("TEST_POSITIVE_INT", 42)

    def test_positive_int_rejects_float(self):
        """Reject float strings."""
        os.environ["TEST_POSITIVE_INT"] = "3.14"
        with self.assertRaises(ValueError):
            env._positive_int("TEST_POSITIVE_INT", 42)


class TestPathResolution(TestCase):
    """Tests for env._path() resolution."""

    def setUp(self):
        """Save original environ."""
        self.original_environ = os.environ.copy()

    def tearDown(self):
        """Restore original environ."""
        os.environ.clear()
        os.environ.update(self.original_environ)

    def test_path_uses_default_when_not_set(self):
        """Return default path when environment variable is not set."""
        os.environ.pop("TEST_PATH", None)
        default = Path("/tmp/default")
        result = env._path("TEST_PATH", default)
        self.assertEqual(result, default.resolve())

    def test_path_resolves_from_environment(self):
        """Use environment variable path when set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["TEST_PATH"] = tmpdir
            result = env._path("TEST_PATH", Path("/tmp/default"))
            self.assertEqual(result, Path(tmpdir).resolve())

    def test_path_expands_user_home(self):
        """Expand ~ to user home directory."""
        os.environ["TEST_PATH"] = "~/test-dir"
        result = env._path("TEST_PATH", Path("/tmp/default"))
        self.assertTrue(str(result).startswith(str(Path.home())))
        self.assertTrue(str(result).endswith("test-dir"))

    def test_path_resolves_relative_path(self):
        """Resolve relative paths to absolute."""
        os.environ["TEST_PATH"] = "./relative/path"
        result = env._path("TEST_PATH", Path("/tmp/default"))
        self.assertTrue(result.is_absolute())

    def test_path_handles_nonexistent_path(self):
        """Resolve nonexistent path without error."""
        os.environ["TEST_PATH"] = "/nonexistent/path/to/dir"
        result = env._path("TEST_PATH", Path("/tmp/default"))
        # Should still resolve to absolute path even if it doesn't exist
        self.assertTrue(result.is_absolute())
        self.assertEqual(str(result), "/nonexistent/path/to/dir")


class TestServerShare(TestCase):
    """Tests for SERVER_SHARE validation."""

    def setUp(self):
        """Save original environ."""
        self.original_environ = os.environ.copy()

    def tearDown(self):
        """Restore original environ."""
        os.environ.clear()
        os.environ.update(self.original_environ)

    def test_server_share_default_is_false(self):
        """Default SHRIMPLY_SERVER_SHARE should be 0 (False)."""
        os.environ.pop("SHRIMPLY_SERVER_SHARE", None)
        # Re-import to get the module-level assignment
        import importlib
        importlib.reload(env)
        self.assertFalse(env.SERVER_SHARE)

    def test_server_share_accepts_zero(self):
        """SHRIMPLY_SERVER_SHARE=0 is valid."""
        os.environ["SHRIMPLY_SERVER_SHARE"] = "0"
        import importlib
        importlib.reload(env)
        self.assertFalse(env.SERVER_SHARE)

    def test_server_share_accepts_one(self):
        """SHRIMPLY_SERVER_SHARE=1 is valid."""
        os.environ["SHRIMPLY_SERVER_SHARE"] = "1"
        import importlib
        importlib.reload(env)
        self.assertTrue(env.SERVER_SHARE)


class TestMinimaxDecodeOffload(TestCase):
    """Tests for minimax_h3_decode_offload() validation."""

    def setUp(self):
        """Save original environ."""
        self.original_environ = os.environ.copy()

    def tearDown(self):
        """Restore original environ."""
        os.environ.clear()
        os.environ.update(self.original_environ)

    def test_decode_offload_default_is_auto(self):
        """Default MINIMAX_H3_DECODE_OFFLOAD should be 'auto'."""
        os.environ.pop("MINIMAX_H3_DECODE_OFFLOAD", None)
        result = env.minimax_h3_decode_offload()
        self.assertEqual(result, "auto")

    def test_decode_offload_accepts_auto(self):
        """MINIMAX_H3_DECODE_OFFLOAD=auto is valid."""
        os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = "auto"
        result = env.minimax_h3_decode_offload()
        self.assertEqual(result, "auto")

    def test_decode_offload_accepts_gpu(self):
        """MINIMAX_H3_DECODE_OFFLOAD=gpu is valid."""
        os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = "gpu"
        result = env.minimax_h3_decode_offload()
        self.assertEqual(result, "gpu")

    def test_decode_offload_accepts_disk(self):
        """MINIMAX_H3_DECODE_OFFLOAD=disk is valid."""
        os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = "disk"
        result = env.minimax_h3_decode_offload()
        self.assertEqual(result, "disk")

    def test_decode_offload_case_insensitive(self):
        """MINIMAX_H3_DECODE_OFFLOAD comparison is case-insensitive."""
        os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = "AUTO"
        result = env.minimax_h3_decode_offload()
        self.assertEqual(result, "auto")

    def test_decode_offload_rejects_invalid_value(self):
        """MINIMAX_H3_DECODE_OFFLOAD must be auto, gpu, or disk."""
        os.environ["MINIMAX_H3_DECODE_OFFLOAD"] = "invalid"
        with self.assertRaises(ValueError) as ctx:
            env.minimax_h3_decode_offload()
        self.assertIn("must be auto, gpu, or disk", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
