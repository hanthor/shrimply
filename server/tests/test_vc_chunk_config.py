"""Unit tests for pneuma voice conversion chunk configuration."""

from unittest import TestCase

from api.pneuma.inference.configs.config import vc_chunk_config


class VCChunkConfigTests(TestCase):
    """Test cases for vc_chunk_config() configuration logic."""

    def test_low_vram_returns_minimal_config(self):
        """Test that ≤4GB VRAM returns minimal chunk configuration."""
        result = vc_chunk_config(is_half=True, gpu_mem=4)
        self.assertEqual(result, (1, 5, 30, 32))

    def test_low_vram_ignores_half_precision(self):
        """Test that ≤4GB VRAM returns minimal config regardless of precision."""
        result_full = vc_chunk_config(is_half=False, gpu_mem=4)
        result_half = vc_chunk_config(is_half=True, gpu_mem=4)
        self.assertEqual(result_full, (1, 5, 30, 32))
        self.assertEqual(result_half, (1, 5, 30, 32))

    def test_very_low_vram_less_than_4gb(self):
        """Test edge case: VRAM < 4GB."""
        result = vc_chunk_config(is_half=False, gpu_mem=2)
        self.assertEqual(result, (1, 5, 30, 32))

    def test_medium_vram_with_half_precision(self):
        """Test that ≥6GB VRAM with half-precision uses balanced config."""
        result = vc_chunk_config(is_half=True, gpu_mem=6)
        self.assertEqual(result, (3, 10, 60, 65))

    def test_medium_vram_with_half_precision_8gb(self):
        """Test 8GB VRAM with half-precision."""
        result = vc_chunk_config(is_half=True, gpu_mem=8)
        self.assertEqual(result, (3, 10, 60, 65))

    def test_medium_vram_without_half_precision(self):
        """Test that ≥4GB VRAM without half-precision uses conservative config."""
        result = vc_chunk_config(is_half=False, gpu_mem=5)
        self.assertEqual(result, (1, 6, 38, 41))

    def test_medium_vram_without_half_precision_6gb(self):
        """Test 6GB VRAM without half-precision uses conservative config."""
        result = vc_chunk_config(is_half=False, gpu_mem=6)
        self.assertEqual(result, (1, 6, 38, 41))

    def test_cpu_only_mode(self):
        """Test CPU-only mode (no GPU memory) uses conservative config."""
        result = vc_chunk_config(is_half=False, gpu_mem=None)
        self.assertEqual(result, (1, 6, 38, 41))

    def test_cpu_only_mode_with_half_precision_flag(self):
        """Test CPU-only mode ignores half-precision flag."""
        result = vc_chunk_config(is_half=True, gpu_mem=None)
        self.assertEqual(result, (1, 6, 38, 41))

    def test_return_value_structure(self):
        """Test that return value is always a 4-tuple of positive integers."""
        configs = [
            vc_chunk_config(is_half=True, gpu_mem=4),
            vc_chunk_config(is_half=True, gpu_mem=6),
            vc_chunk_config(is_half=False, gpu_mem=5),
            vc_chunk_config(is_half=False, gpu_mem=None),
        ]
        
        for config in configs:
            self.assertIsInstance(config, tuple)
            self.assertEqual(len(config), 4)
            for value in config:
                self.assertIsInstance(value, int)
                self.assertGreater(value, 0)

    def test_chunk_ordering_invariant(self):
        """Test that chunk sizes follow expected ordering: x_pad ≤ x_query < x_center < x_max."""
        test_cases = [
            (True, 4),
            (True, 6),
            (False, 5),
            (False, None),
        ]
        
        for is_half, gpu_mem in test_cases:
            x_pad, x_query, x_center, x_max = vc_chunk_config(is_half, gpu_mem)
            self.assertLessEqual(x_pad, x_query, 
                f"x_pad ({x_pad}) should be ≤ x_query ({x_query}) for is_half={is_half}, gpu_mem={gpu_mem}")
            self.assertLess(x_query, x_center,
                f"x_query ({x_query}) should be < x_center ({x_center}) for is_half={is_half}, gpu_mem={gpu_mem}")
            self.assertLess(x_center, x_max,
                f"x_center ({x_center}) should be < x_max ({x_max}) for is_half={is_half}, gpu_mem={gpu_mem}")

    def test_deterministic_behavior(self):
        """Test that function returns same result on repeated calls."""
        for _ in range(3):
            result1 = vc_chunk_config(is_half=True, gpu_mem=6)
            result2 = vc_chunk_config(is_half=True, gpu_mem=6)
            self.assertEqual(result1, result2)
        
        for _ in range(3):
            result1 = vc_chunk_config(is_half=False, gpu_mem=None)
            result2 = vc_chunk_config(is_half=False, gpu_mem=None)
            self.assertEqual(result1, result2)

    def test_boundary_4gb_threshold(self):
        """Test boundary condition at 4GB threshold."""
        result_3gb = vc_chunk_config(is_half=False, gpu_mem=3)
        result_4gb = vc_chunk_config(is_half=False, gpu_mem=4)
        result_5gb = vc_chunk_config(is_half=False, gpu_mem=5)
        
        # ≤4GB should use minimal config
        self.assertEqual(result_3gb, (1, 5, 30, 32))
        self.assertEqual(result_4gb, (1, 5, 30, 32))
        
        # >4GB without half-precision should use conservative config
        self.assertEqual(result_5gb, (1, 6, 38, 41))

    def test_boundary_6gb_threshold_with_half_precision(self):
        """Test boundary condition at 6GB threshold with half-precision."""
        # 5GB with half-precision still uses conservative (not balanced) config
        result_5gb_half = vc_chunk_config(is_half=True, gpu_mem=5)
        self.assertEqual(result_5gb_half, (1, 6, 38, 41))
        
        # 6GB with half-precision uses balanced config
        result_6gb_half = vc_chunk_config(is_half=True, gpu_mem=6)
        self.assertEqual(result_6gb_half, (3, 10, 60, 65))

    def test_large_vram_uses_appropriate_config(self):
        """Test that large VRAM amounts use appropriate config based on precision."""
        result_half = vc_chunk_config(is_half=True, gpu_mem=24)
        result_no_half = vc_chunk_config(is_half=False, gpu_mem=24)
        
        self.assertEqual(result_half, (3, 10, 60, 65))
        self.assertEqual(result_no_half, (1, 6, 38, 41))
