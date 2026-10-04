"""Tests for pneuma inference configuration utilities."""

import sys
import os
from unittest import TestCase

# Add server/src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from api.pneuma.inference.configs.config import vc_chunk_config


class VcChunkConfigTests(TestCase):
    """Test cases for vc_chunk_config() voice conversion chunk configuration."""

    # Low VRAM cases (4GB or less)
    def test_low_vram_4gb_or_less_returns_minimal_chunks(self):
        """Low VRAM (4GB or less) should use minimal chunk sizes."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=False, gpu_mem=4)
        self.assertEqual((x_pad, x_query, x_center, x_max), (1, 5, 30, 32))

    def test_low_vram_2gb_returns_minimal_chunks(self):
        """Very low VRAM (2GB) should use minimal chunk sizes."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=False, gpu_mem=2)
        self.assertEqual((x_pad, x_query, x_center, x_max), (1, 5, 30, 32))

    def test_low_vram_1gb_returns_minimal_chunks(self):
        """Minimal VRAM (1GB) should use minimal chunk sizes."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=False, gpu_mem=1)
        self.assertEqual((x_pad, x_query, x_center, x_max), (1, 5, 30, 32))

    def test_low_vram_with_half_precision_still_minimal(self):
        """Low VRAM with half-precision should still use minimal chunk sizes."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=True, gpu_mem=4)
        self.assertEqual((x_pad, x_query, x_center, x_max), (1, 5, 30, 32))

    # Medium VRAM with half-precision (6GB+)
    def test_medium_vram_half_precision_returns_balanced_chunks(self):
        """Medium VRAM (6GB) with half-precision should use balanced chunk sizes."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=True, gpu_mem=6)
        self.assertEqual((x_pad, x_query, x_center, x_max), (3, 10, 60, 65))

    def test_medium_vram_8gb_half_precision_returns_balanced_chunks(self):
        """Medium VRAM (8GB) with half-precision should use balanced chunk sizes."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=True, gpu_mem=8)
        self.assertEqual((x_pad, x_query, x_center, x_max), (3, 10, 60, 65))

    def test_high_vram_12gb_half_precision_returns_balanced_chunks(self):
        """High VRAM (12GB) with half-precision should use balanced chunk sizes."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=True, gpu_mem=12)
        self.assertEqual((x_pad, x_query, x_center, x_max), (3, 10, 60, 65))

    def test_very_high_vram_24gb_half_precision_returns_balanced_chunks(self):
        """Very high VRAM (24GB) with half-precision should use balanced chunk sizes."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=True, gpu_mem=24)
        self.assertEqual((x_pad, x_query, x_center, x_max), (3, 10, 60, 65))

    # Medium VRAM without half-precision (4GB+)
    def test_medium_vram_full_precision_returns_conservative_chunks(self):
        """Medium VRAM (4GB) without half-precision should use conservative chunks."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=False, gpu_mem=5)
        self.assertEqual((x_pad, x_query, x_center, x_max), (1, 6, 38, 41))

    def test_high_vram_8gb_full_precision_returns_conservative_chunks(self):
        """High VRAM (8GB) without half-precision should use conservative chunks."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=False, gpu_mem=8)
        self.assertEqual((x_pad, x_query, x_center, x_max), (1, 6, 38, 41))

    def test_very_high_vram_24gb_full_precision_returns_conservative_chunks(self):
        """Very high VRAM (24GB) without half-precision should use conservative chunks."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=False, gpu_mem=24)
        self.assertEqual((x_pad, x_query, x_center, x_max), (1, 6, 38, 41))

    # CPU-only (no GPU memory)
    def test_cpu_only_no_gpu_memory_returns_conservative_chunks(self):
        """CPU-only mode (no GPU memory) should use conservative chunks."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=False, gpu_mem=None)
        self.assertEqual((x_pad, x_query, x_center, x_max), (1, 6, 38, 41))

    def test_cpu_only_with_half_precision_returns_conservative_chunks(self):
        """CPU-only with half-precision flag should still use conservative chunks."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=True, gpu_mem=None)
        self.assertEqual((x_pad, x_query, x_center, x_max), (1, 6, 38, 41))

    # Boundary conditions
    def test_boundary_4gb_exact_is_low_vram(self):
        """Exactly 4GB should be classified as low VRAM."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=False, gpu_mem=4)
        self.assertEqual((x_pad, x_query, x_center, x_max), (1, 5, 30, 32))

    def test_boundary_5gb_is_not_low_vram(self):
        """5GB should NOT be classified as low VRAM (above threshold)."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=False, gpu_mem=5)
        # Should return conservative (not minimal) chunks
        self.assertEqual((x_pad, x_query, x_center, x_max), (1, 6, 38, 41))

    def test_boundary_5gb_with_half_precision_is_medium_vram(self):
        """5GB with half-precision should qualify for balanced chunks (needs 6GB check)."""
        # Note: 5GB with half is still below 6GB threshold, so should use conservative
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=True, gpu_mem=5)
        # Falls through to the full-precision case (> 4GB but not >= 6GB with half)
        self.assertEqual((x_pad, x_query, x_center, x_max), (1, 6, 38, 41))

    def test_boundary_6gb_with_half_precision_is_medium_vram(self):
        """6GB with half-precision should qualify for balanced chunks."""
        x_pad, x_query, x_center, x_max = vc_chunk_config(is_half=True, gpu_mem=6)
        self.assertEqual((x_pad, x_query, x_center, x_max), (3, 10, 60, 65))

    # Return value structure validation
    def test_returns_tuple_of_four_integers(self):
        """Should always return a tuple of four integers."""
        result = vc_chunk_config(is_half=False, gpu_mem=8)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        for value in result:
            self.assertIsInstance(value, int)

    def test_chunk_values_are_positive(self):
        """All chunk values should be positive integers."""
        for is_half in [True, False]:
            for gpu_mem in [None, 1, 4, 6, 12, 24]:
                x_pad, x_query, x_center, x_max = vc_chunk_config(is_half, gpu_mem)
                self.assertGreater(x_pad, 0)
                self.assertGreater(x_query, 0)
                self.assertGreater(x_center, 0)
                self.assertGreater(x_max, 0)

    def test_chunk_ordering_x_max_greater_than_x_center(self):
        """x_max should always be greater than or equal to x_center."""
        for is_half in [True, False]:
            for gpu_mem in [None, 1, 4, 6, 12, 24]:
                x_pad, x_query, x_center, x_max = vc_chunk_config(is_half, gpu_mem)
                self.assertGreaterEqual(x_max, x_center)

    def test_chunk_ordering_x_center_greater_than_x_query(self):
        """x_center should generally be greater than x_query."""
        for is_half in [True, False]:
            for gpu_mem in [None, 1, 4, 6, 12, 24]:
                x_pad, x_query, x_center, x_max = vc_chunk_config(is_half, gpu_mem)
                self.assertGreater(x_center, x_query)

    def test_deterministic_results(self):
        """Same inputs should always produce identical results."""
        result1 = vc_chunk_config(is_half=True, gpu_mem=8)
        result2 = vc_chunk_config(is_half=True, gpu_mem=8)
        result3 = vc_chunk_config(is_half=True, gpu_mem=8)
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)
