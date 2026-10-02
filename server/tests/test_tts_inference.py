import importlib.util
import sys
import types
from fractions import Fraction
from types import SimpleNamespace
from unittest import SkipTest, TestCase
from unittest.mock import patch

# WeTextProcessing needs pynini; the text normalizers are unused here because
# the tokenizer is faked.
if importlib.util.find_spec("tn") is None:
    for name in ("tn", "tn.chinese", "tn.english"):
        sys.modules[name] = types.ModuleType(name)
    for name in ("tn.chinese.normalizer", "tn.english.normalizer"):
        module = types.ModuleType(name)
        module.Normalizer = object
        sys.modules[name] = module

try:
    import torch
    import torchaudio  # noqa: F401
    from api.tts.index_tts_2_0 import inference
except ImportError as error:
    raise SkipTest(f"TTS inference dependencies unavailable: {error}") from error

from api.tts.protocol import Audio
from api.tts.requests import (
    AudioEmotion,
    DurationTiming,
    EmotionFactors,
    FactorEmotion,
    IndexSampling,
    IndexTts2Request,
    SpeakerEmotion,
    SpeedTiming,
    TextEmotion,
)


WAV = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 36
HOP = 256
SAMPLE_RATE = 22_050
NATURAL_FRAMES = 100
BASE = torch.tensor([[2.0, 2.0, 2.0]])


def sampling() -> IndexSampling:
    return IndexSampling(
        do_sample=False,
        typical_sampling=False,
        typical_mass=0.9,
        top_p=0.8,
        top_k=30,
        temperature=1.0,
        length_penalty=0.0,
        num_beams=1,
        repetition_penalty=10.0,
        max_mel_tokens=1500,
    )


def factors(**values: float) -> EmotionFactors:
    names = (
        "happy",
        "angry",
        "sad",
        "afraid",
        "disgusted",
        "melancholic",
        "surprised",
        "calm",
    )
    return EmotionFactors(**{name: values.get(name, 0.0) for name in names})


def request(
    emotion: object | None = None,
    timing: object | None = None,
    text: str = "one two",
    max_tokens: int = 100,
    silence: Fraction = Fraction(0),
) -> IndexTts2Request:
    return IndexTts2Request(
        model="IndexTeam/IndexTTS-2",
        text=text,
        voice=Audio(wav=WAV),
        emotion=emotion or SpeakerEmotion(),
        timing=timing or SpeedTiming(factor=Fraction(1)),
        sampling=sampling(),
        max_text_tokens_per_segment=max_tokens,
        intersegment_silence=silence,
        glossary=[],
    )


class FakeTokenizer:
    def __init__(self, empty: bool = False) -> None:
        self.empty = empty

    def tokenize(self, text: str, glossary: list[object]) -> list[str]:
        return text.split()

    def split_segments(self, tokens: list[str], maximum: int) -> list[list[str]]:
        if self.empty:
            return []
        return [tokens[i : i + maximum] for i in range(0, len(tokens), maximum)]


class FakeGpt:
    def __init__(self) -> None:
        self.merge_calls: list[float] = []

    def emotion_vector(
        self, conditioning: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        return BASE.clone()

    def merge_emotion_vectors(
        self,
        speaker: torch.Tensor,
        emotion: torch.Tensor,
        speaker_lengths: torch.Tensor,
        emotion_lengths: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        self.merge_calls.append(strength)
        return BASE * strength


def fake_models(empty_tokenizer: bool = False) -> SimpleNamespace:
    # Group g has two prototypes; the speaker prototypes make index 1 the
    # nearest match for a style of [0, 1].
    emotion_groups = tuple(
        torch.tensor([[g + 1.0, 0.0, 0.0], [0.0, g + 1.0, 1.0]]) for g in range(8)
    )
    speaker_group = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        gpt=FakeGpt(),
        emotion_prototypes=emotion_groups,
        speaker_prototypes=(speaker_group,) * 8,
        emotion_text_model="unused",
        tokenizer=FakeTokenizer(empty_tokenizer),
    )


def fake_reference(prompt_frames: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        conditioning=torch.zeros(1, 5, 3),
        prompt_condition=torch.zeros(1, prompt_frames, 4),
        mel=torch.zeros(1, 80, 10),
        style=torch.tensor([[0.0, 1.0]]),
    )


class NormalizeEmotionFactorsTests(TestCase):
    def test_requires_exactly_eight_factors(self) -> None:
        for count in (0, 4, 9):
            with self.assertRaisesRegex(ValueError, "eight emotion factors"):
                inference._normalize_emotion_factors([0.1] * count)

    def test_applies_bias_when_total_is_within_limit(self) -> None:
        result = inference._normalize_emotion_factors([0.4] + [0.0] * 7)
        self.assertAlmostEqual(result[0], 0.4 * 0.9375)
        self.assertEqual(result[1:], [0.0] * 7)

    def test_clamps_negative_values_to_zero(self) -> None:
        result = inference._normalize_emotion_factors([-1.0, 0.2] + [0.0] * 6)
        self.assertEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.2 * 0.875)

    def test_rescales_total_to_cap_and_keeps_ratios(self) -> None:
        result = inference._normalize_emotion_factors([0.8] * 8)
        self.assertAlmostEqual(sum(result), 0.8)
        self.assertAlmostEqual(result[0] / result[7], 0.9375 / 0.5625)


class PrototypeEmotionVectorTests(TestCase):
    def test_picks_nearest_speaker_prototype_per_group(self) -> None:
        models = fake_models()
        vector = inference._prototype_emotion_vector(
            models, torch.tensor([[0.0, 1.0]]), BASE, [0.5] + [0.0] * 7, False
        )
        # 0.5 * emotion_group_0[1] + (1 - 0.5) * base
        self.assertTrue(torch.allclose(vector, torch.tensor([[1.0, 1.5, 1.5]])))

    def test_zero_factors_return_base(self) -> None:
        vector = inference._prototype_emotion_vector(
            fake_models(), torch.tensor([[0.0, 1.0]]), BASE, [0.0] * 8, False
        )
        self.assertTrue(torch.equal(vector, BASE))

    def test_randomize_ignores_style_and_uses_random_index(self) -> None:
        with patch.object(inference.random, "randrange", return_value=0) as pick:
            vector = inference._prototype_emotion_vector(
                fake_models(), torch.tensor([[0.0, 1.0]]), BASE, [0.5] + [0.0] * 7, True
            )
        self.assertEqual(pick.call_count, 8)
        self.assertTrue(torch.allclose(vector, torch.tensor([[1.5, 1.0, 1.0]])))


class EmotionConditionTests(TestCase):
    def test_speaker_emotion_uses_reference_and_base_vector(self) -> None:
        reference = fake_reference()
        condition = inference._emotion_condition(
            fake_models(), request(SpeakerEmotion()), reference
        )
        self.assertIs(condition.conditioning, reference.conditioning)
        self.assertTrue(torch.equal(condition.vector, BASE))

    def test_factor_emotion_scales_normalized_factors_by_strength(self) -> None:
        emotion = FactorEmotion(
            factors=factors(happy=0.4), strength=0.5, randomize=False
        )
        reference = fake_reference()
        condition = inference._emotion_condition(
            fake_models(), request(emotion), reference
        )
        scaled = 0.4 * 0.9375 * 0.5
        expected = scaled * torch.tensor([[0.0, 1.0, 1.0]]) + (1 - scaled) * BASE
        self.assertIs(condition.conditioning, reference.conditioning)
        self.assertTrue(torch.allclose(condition.vector, expected))

    def test_audio_emotion_merges_with_strength(self) -> None:
        models = fake_models()
        emotion_conditioning = torch.zeros(1, 7, 3)
        emotion = AudioEmotion(audio=Audio(wav=WAV), strength=0.25)
        with patch.object(
            inference, "_emotion_audio_conditioning", return_value=emotion_conditioning
        ) as audio_conditioning:
            condition = inference._emotion_condition(
                models, request(emotion), fake_reference()
            )
        audio_conditioning.assert_called_once_with(models, WAV)
        self.assertIs(condition.conditioning, emotion_conditioning)
        self.assertEqual(models.gpt.merge_calls, [0.25])
        self.assertTrue(torch.equal(condition.vector, BASE * 0.25))

    def test_text_emotion_falls_back_to_request_text(self) -> None:
        seen: list[str] = []

        def analyze(text: str) -> list[float]:
            seen.append(text)
            return [0.4] + [0.0] * 7

        analyzer = SimpleNamespace(analyze=analyze)
        with patch.object(
            inference, "load_emotion_text_analyzer", return_value=analyzer
        ):
            inference._emotion_condition(
                fake_models(),
                request(TextEmotion(text=None, strength=1.0, randomize=False)),
                fake_reference(),
            )
            inference._emotion_condition(
                fake_models(),
                request(TextEmotion(text="angry", strength=1.0, randomize=False)),
                fake_reference(),
            )
        self.assertEqual(seen, ["one two", "angry"])

    def test_text_emotion_rejects_wrong_factor_count(self) -> None:
        analyzer = SimpleNamespace(analyze=lambda text: [0.1] * 3)
        with patch.object(
            inference, "load_emotion_text_analyzer", return_value=analyzer
        ):
            with self.assertRaisesRegex(ValueError, "eight emotion factors"):
                inference._emotion_condition(
                    fake_models(),
                    request(TextEmotion(text=None, strength=1.0, randomize=False)),
                    fake_reference(),
                )


class SynthesizeTests(TestCase):
    """Orchestration of synthesize(); the neural stages are replaced by fakes."""

    def run_synthesize(
        self,
        req: IndexTts2Request,
        prompt_frames: int = 10,
        value: float = 0.25,
        empty_tokenizer: bool = False,
        progress: object | None = None,
    ) -> inference.SynthesisOutput:
        rendered_frames: list[int] = []

        def render(models, embeddings, target_frames, reference, timings):
            rendered_frames.append(target_frames)
            return torch.full((1, target_frames * HOP), value)

        self.rendered_frames = rendered_frames
        with (
            patch.object(
                inference,
                "_reference_features",
                return_value=fake_reference(prompt_frames),
            ),
            patch.object(
                inference,
                "_emotion_condition",
                return_value=inference.EmotionCondition(torch.zeros(1, 5, 3), BASE),
            ),
            patch.object(
                inference,
                "_semantic_embeddings",
                return_value=(torch.zeros(1, 4, 4), NATURAL_FRAMES),
            ),
            patch.object(inference, "_render", side_effect=render),
        ):
            return inference.synthesize(
                fake_models(empty_tokenizer), req, progress=progress
            )

    def test_speed_one_keeps_natural_length(self) -> None:
        output = self.run_synthesize(request(text="one"))
        self.assertEqual(output.sample_rate, SAMPLE_RATE)
        self.assertEqual(output.waveform.shape, (1, NATURAL_FRAMES * HOP))
        self.assertEqual(output.speed_factor, Fraction(1))

    def test_speed_factor_scales_rendered_frames(self) -> None:
        output = self.run_synthesize(
            request(text="one", timing=SpeedTiming(factor=Fraction(2)))
        )
        self.assertEqual(self.rendered_frames, [NATURAL_FRAMES // 2])
        self.assertEqual(output.waveform.shape, (1, NATURAL_FRAMES // 2 * HOP))
        self.assertEqual(output.speed_factor, Fraction(2))

    def test_segments_are_joined_with_silence(self) -> None:
        output = self.run_synthesize(
            request(max_tokens=1, silence=Fraction(1, 2)),
        )
        silence = round(0.5 * SAMPLE_RATE)
        segment = NATURAL_FRAMES * HOP
        self.assertEqual(output.waveform.shape, (1, 2 * segment + silence))
        self.assertTrue(
            torch.equal(
                output.waveform[0, segment : segment + silence],
                torch.zeros(silence),
            )
        )
        self.assertEqual(output.speed_factor, Fraction(1))

    def test_no_silence_after_last_segment(self) -> None:
        output = self.run_synthesize(request(max_tokens=1, silence=Fraction(0)))
        self.assertEqual(output.waveform.shape, (1, 2 * NATURAL_FRAMES * HOP))

    def test_output_is_clamped_to_unit_range(self) -> None:
        output = self.run_synthesize(request(text="one"), value=3.0)
        self.assertEqual(float(output.waveform.max()), 1.0)

    def test_duration_matching_natural_frames_skips_warping(self) -> None:
        seconds = Fraction(NATURAL_FRAMES * HOP, SAMPLE_RATE)
        output = self.run_synthesize(
            request(text="one", timing=DurationTiming(seconds=seconds))
        )
        self.assertEqual(self.rendered_frames, [NATURAL_FRAMES])
        self.assertEqual(output.waveform.shape, (1, NATURAL_FRAMES * HOP))

    def test_duration_differing_from_natural_renders_target_frames(self) -> None:
        target = 50
        seconds = Fraction(target * HOP, SAMPLE_RATE)
        output = self.run_synthesize(
            request(text="one", timing=DurationTiming(seconds=seconds))
        )
        self.assertEqual(self.rendered_frames, [NATURAL_FRAMES, target])
        self.assertEqual(output.waveform.shape, (1, target * HOP))
        self.assertEqual(output.speed_factor, Fraction(NATURAL_FRAMES, target))

    def test_duration_over_sequence_limit_is_rejected(self) -> None:
        seconds = Fraction(300 * HOP, SAMPLE_RATE)
        with self.assertRaisesRegex(ValueError, "Duration exceeds"):
            self.run_synthesize(
                request(text="one", timing=DurationTiming(seconds=seconds)),
                prompt_frames=8_000,
            )

    def test_speed_over_sequence_limit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Speed factor exceeds"):
            self.run_synthesize(
                request(text="one", timing=SpeedTiming(factor=Fraction(1, 2))),
                prompt_frames=8_000,
            )

    def test_text_without_tokens_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no IndexTTS tokens"):
            self.run_synthesize(request(text="one"), empty_tokenizer=True)

    def test_progress_reports_each_stage_in_order(self) -> None:
        events: list[tuple[float, str]] = []
        self.run_synthesize(
            request(max_tokens=1),
            progress=lambda fraction, label: events.append((fraction, label)),
        )
        self.assertEqual(events[0], (0.05, "Preparing reference voice"))
        self.assertEqual(events[1][1], "Synthesizing segment 1 of 2")
        self.assertEqual(events[2][1], "Synthesizing segment 2 of 2")
        self.assertEqual(events[-1], (0.95, "Finalizing speech"))
        fractions = [fraction for fraction, _ in events]
        self.assertEqual(fractions, sorted(fractions))
