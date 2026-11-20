"""Unit tests for pure tensor math in api.pneuma.inference.f0.gen.

These cover the parts of the F0 (pitch detection) pipeline that are plain
tensor arithmetic and carry no model/network dependency: linear
interpolation, post-processing (pitch shifting, manual F0 override, coarse
pitch quantization), voiced-frame gap filling, and F0 resampling.

Model loading (`load_f0_model`) and the model-dependent branches of
`extract_f0` (crepe/rmvpe/fcpe/swift-f0 inference) are out of scope here —
they require heavy model weights and are not exercised by these tests.
"""

from math import log

import torch

from api.pneuma.inference.f0.gen import (
    _interpolate_f0,
    _linear_interpolate,
    _resize_f0,
    _to_float32_tensor,
    post_process,
)

F0_MIN = 50
F0_MAX = 1100
F0_MEL_MIN = 1127 * log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * log(1 + F0_MAX / 700)


class TestLinearInterpolate:
    def test_interpolates_between_known_points(self) -> None:
        xp = torch.tensor([0.0, 1.0, 2.0])
        fp = torch.tensor([10.0, 20.0, 30.0])
        x = torch.tensor([0.5, 1.5])

        result = _linear_interpolate(x, xp, fp)

        assert torch.allclose(result, torch.tensor([15.0, 25.0]))

    def test_clamps_below_first_known_point(self) -> None:
        xp = torch.tensor([0.0, 1.0, 2.0])
        fp = torch.tensor([10.0, 20.0, 30.0])

        result = _linear_interpolate(torch.tensor([-5.0]), xp, fp)

        assert torch.allclose(result, torch.tensor([10.0]))

    def test_clamps_above_last_known_point(self) -> None:
        xp = torch.tensor([0.0, 1.0, 2.0])
        fp = torch.tensor([10.0, 20.0, 30.0])

        result = _linear_interpolate(torch.tensor([99.0]), xp, fp)

        assert torch.allclose(result, torch.tensor([30.0]))

    def test_empty_known_points_returns_zeros(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0])

        result = _linear_interpolate(x, torch.tensor([]), torch.tensor([]))

        assert torch.equal(result, torch.zeros(3, dtype=torch.float32))

    def test_single_known_point_broadcasts_its_value(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0])

        result = _linear_interpolate(x, torch.tensor([5.0]), torch.tensor([42.0]))

        assert torch.equal(result, torch.full((3,), 42.0))

    def test_returned_dtype_is_float32(self) -> None:
        xp = torch.tensor([0.0, 1.0], dtype=torch.float64)
        fp = torch.tensor([1.0, 2.0], dtype=torch.float64)
        x = torch.tensor([0.5], dtype=torch.float64)

        result = _linear_interpolate(x, xp, fp)

        assert result.dtype == torch.float32


class TestToFloat32Tensor:
    def test_converts_nested_list(self) -> None:
        result = _to_float32_tensor([[1.0, 2.0], [3.0, 4.0]])

        assert result.dtype == torch.float32
        assert torch.equal(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_converts_existing_tensor_and_dtype(self) -> None:
        source = torch.tensor([1, 2, 3], dtype=torch.int64)

        result = _to_float32_tensor(source)

        assert result.dtype == torch.float32
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_preserves_values_when_already_float32(self) -> None:
        source = torch.tensor([1.5, -2.5], dtype=torch.float32)

        result = _to_float32_tensor(source)

        assert torch.equal(result, source)


class TestInterpolateF0:
    def test_fills_gaps_between_voiced_frames(self) -> None:
        f0 = torch.tensor([0.0, 100.0, 0.0, 200.0, 0.0])

        result = _interpolate_f0(f0)

        # unvoiced (0) frames get filled from surrounding voiced frames;
        # the interior gap is a linear ramp, and edges clamp to the nearest
        # voiced value.
        assert torch.allclose(result, torch.tensor([100.0, 100.0, 150.0, 200.0, 200.0]))

    def test_all_unvoiced_returns_zeros(self) -> None:
        f0 = torch.zeros(4)

        result = _interpolate_f0(f0)

        assert torch.equal(result, torch.zeros(4))

    def test_all_voiced_is_unchanged(self) -> None:
        f0 = torch.tensor([100.0, 110.0, 120.0])

        result = _interpolate_f0(f0)

        assert torch.allclose(result, f0)

    def test_flattens_multi_dimensional_input(self) -> None:
        f0 = torch.tensor([[100.0, 0.0, 200.0]])

        result = _interpolate_f0(f0)

        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([100.0, 150.0, 200.0]))


class TestResizeF0:
    def test_upsamples_to_longer_target(self) -> None:
        source = torch.tensor([100.0, 200.0])

        result = _resize_f0(source, 4)

        assert result.shape == (4,)
        assert torch.allclose(result, torch.tensor([100.0, 150.0, 200.0, 200.0]))

    def test_target_len_zero_returns_empty(self) -> None:
        result = _resize_f0(torch.tensor([100.0, 200.0]), 0)

        assert result.numel() == 0

    def test_empty_source_returns_zeros_at_target_len(self) -> None:
        result = _resize_f0(torch.tensor([]), 3)

        assert torch.equal(result, torch.zeros(3))

    def test_near_silent_values_are_treated_as_unvoiced(self) -> None:
        # values below 0.001 are masked to NaN internally, then filled back
        # in via nan_to_num after interpolation -- verify the result stays
        # finite and doesn't just propagate NaN.
        source = torch.tensor([0.0, 200.0, 0.0])

        result = _resize_f0(source, 3)

        assert torch.isfinite(result).all()


class TestPostProcess:
    def test_returns_coarse_and_continuous_pitch(self) -> None:
        f0 = torch.tensor([0.0, 100.0, 300.0, 800.0])

        coarse, continuous = post_process(100, f0, 0, 3, F0_MEL_MIN, F0_MEL_MAX)

        assert coarse.dtype == torch.int32
        assert continuous.dtype == torch.float32
        assert torch.equal(continuous, f0)

    def test_coarse_pitch_stays_within_embedding_range(self) -> None:
        f0 = torch.tensor([0.0, 50.0, 100.0, 500.0, 1100.0, 5000.0])

        coarse, _ = post_process(100, f0, 0, 3, F0_MEL_MIN, F0_MEL_MAX)

        assert int(coarse.min()) >= 0
        assert int(coarse.max()) <= 255

    def test_pitch_shift_scales_continuous_f0_by_semitones(self) -> None:
        f0 = torch.tensor([100.0, 200.0])

        _, unshifted = post_process(100, f0, 0, 3, F0_MEL_MIN, F0_MEL_MAX)
        _, shifted_up_octave = post_process(100, f0, 12, 3, F0_MEL_MIN, F0_MEL_MAX)

        # +12 semitones is exactly one octave => doubled frequency
        assert torch.allclose(shifted_up_octave, unshifted * 2, rtol=1e-4)

    def test_manual_f0_overrides_a_time_window(self) -> None:
        f0 = torch.zeros(50)
        # manual_f0 rows are (time_seconds, frequency_hz)
        manual = torch.tensor([[0.0, 5.0], [0.02, 8.0]])

        _, continuous = post_process(
            100, f0, 0, 0, F0_MEL_MIN, F0_MEL_MAX, manual_f0=manual
        )

        # the overridden window ramps from 5.0 toward 8.0 rather than
        # staying at the original all-zero input
        assert continuous[0].item() != 0.0
        assert continuous[0].item() == 5.0

    def test_manual_f0_with_coincident_points_replaces_one_frame(self) -> None:
        # a zero time span still rounds up to a 1-frame delta_t (+1 in the
        # implementation), so exactly the first frame is replaced and the
        # rest of f0 is untouched.
        f0 = torch.tensor([42.0, 42.0, 42.0])
        manual = torch.tensor([[0.0, 5.0], [0.0, 5.0]])

        _, continuous = post_process(
            100, f0, 0, 0, F0_MEL_MIN, F0_MEL_MAX, manual_f0=manual
        )

        assert continuous[0].item() == 5.0
        assert torch.equal(continuous[1:], f0[1:])

    def test_empty_f0_does_not_raise(self) -> None:
        coarse, continuous = post_process(
            100, torch.tensor([]), 0, 3, F0_MEL_MIN, F0_MEL_MAX
        )

        assert coarse.numel() == 0
        assert continuous.numel() == 0
