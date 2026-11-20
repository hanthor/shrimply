import importlib.util
import sys
import types
from fractions import Fraction
from unittest import TestCase

if importlib.util.find_spec("torch") is None:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch_stub

from pydantic import ValidationError

from api.tts.protocol import (
    Audio,
    AudioValue,
    ModelInfo,
    NumberInput,
    NumberValue,
    Rational,
    SelectInput,
    SelectOption,
    SelectValue,
    Speech,
    SynthesisRequest,
    TableInput,
    TableValue,
    TextInput,
    TextValue,
    ToggleInput,
    ToggleValue,
    VisibleWhen,
)
from api.tts.requests import (
    DurationTiming,
    FactorEmotion,
    IndexTts2Request,
    QwenCustomVoiceRequest,
    QwenVoiceCloneRequest,
    QwenVoiceDesignRequest,
    SpeedTiming,
    TextEmotion,
    audio,
    integer,
    parse_request,
    request_precision,
    request_text,
    selection,
    text,
    toggle,
    validate_inputs,
)


WAV = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 36


def rational(numerator: int, denominator: int = 1) -> Rational:
    return Rational(numerator=numerator, denominator=denominator)


def text_value(value: str) -> TextValue:
    return TextValue(kind="text", value=value)


def selection_value(value: str) -> SelectValue:
    return SelectValue(kind="select", value=value)


def toggle_value(value: bool) -> ToggleValue:
    return ToggleValue(kind="toggle", value=value)


def number_value(numerator: int, denominator: int = 1) -> NumberValue:
    return NumberValue(kind="number", value=rational(numerator, denominator))


def audio_value() -> AudioValue:
    return AudioValue(kind="audio", value=Audio(wav=WAV))


def request(model: str, **inputs: object) -> SynthesisRequest:
    return SynthesisRequest(model=model, inputs=inputs)


class ProtocolTests(TestCase):
    def test_audio_requires_wav_header(self) -> None:
        self.assertEqual(Audio(wav=WAV).wav, WAV)
        with self.assertRaisesRegex(ValidationError, "Audio must contain WAV data"):
            Audio(wav=b"not wav" + b"\x00" * 40)

    def test_protocol_models_are_strict(self) -> None:
        with self.assertRaises(ValidationError):
            Rational(numerator="1", denominator=1)
        with self.assertRaises(ValidationError):
            TextValue(kind="text", value=1)

    def test_select_input_rejects_duplicate_or_missing_default(self) -> None:
        with self.assertRaisesRegex(ValidationError, "duplicate options"):
            SelectInput(
                key="mode",
                label="Mode",
                options=[
                    SelectOption(value="one", label="One"),
                    SelectOption(value="one", label="Duplicate"),
                ],
                default="one",
            )
        with self.assertRaisesRegex(ValidationError, "Default option"):
            SelectInput(
                key="mode",
                label="Mode",
                options=[SelectOption(value="one", label="One")],
                default="two",
            )

    def test_number_input_rejects_invalid_step_and_default_range(self) -> None:
        with self.assertRaisesRegex(ValidationError, "step must be positive"):
            NumberInput(
                key="amount",
                label="Amount",
                default=rational(1),
                minimum=rational(0),
                maximum=rational(2),
                step=rational(0),
            )
        with self.assertRaisesRegex(ValidationError, "default must be within"):
            NumberInput(
                key="amount",
                label="Amount",
                default=rational(3),
                minimum=rational(0),
                maximum=rational(2),
                step=rational(1),
            )

    def test_table_input_rejects_duplicate_columns(self) -> None:
        with self.assertRaisesRegex(ValidationError, "duplicate table columns"):
            TableInput(
                key="glossary",
                label="Glossary",
                columns=[
                    {"key": "term", "label": "Term"},
                    {"key": "term", "label": "Duplicate"},
                ],
            )

    def test_model_info_rejects_invalid_visibility_controller(self) -> None:
        with self.assertRaisesRegex(ValidationError, "missing or invalid controller"):
            ModelInfo(
                id="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                label="Qwen",
                inputs=[
                    TextInput(key="text", label="Text"),
                    TextInput(
                        key="dependent",
                        label="Dependent",
                        visible_when=VisibleWhen(input="text", values=["x"]),
                    ),
                ],
            )

    def test_speech_rejects_nonpositive_speed(self) -> None:
        speech = Speech(wav=WAV)
        self.assertEqual(speech.speed_factor, rational(1))
        with self.assertRaisesRegex(ValidationError, "speed factor must be positive"):
            Speech(wav=WAV, speed_factor=rational(0))


class TtsRequestTests(TestCase):
    def qwen_custom_inputs(self, *, instruction: str | None = None) -> dict[str, object]:
        inputs: dict[str, object] = {
            "precision": selection_value("auto"),
            "text": text_value("Hello"),
            "language": selection_value("English"),
            "speaker": selection_value("Vivian"),
        }
        if instruction is not None:
            inputs["instruction"] = text_value(instruction)
        return inputs

    def qwen_clone_inputs(self, reference_text: str) -> dict[str, object]:
        return {
            "precision": selection_value("float16"),
            "text": text_value("Clone this voice"),
            "language": selection_value("English"),
            "reference_audio": audio_value(),
            "reference_text": text_value(reference_text),
        }

    def index_inputs(
        self,
        *,
        model: str = "IndexTeam/IndexTTS-2",
        emotion_mode: str = "speaker",
        timing_mode: str = "speed_factor",
        do_sample: bool = False,
    ) -> dict[str, object]:
        inputs: dict[str, object] = {
            "precision": selection_value("auto"),
            "text": text_value("Index this"),
            "reference_audio": audio_value(),
            "emotion_mode": selection_value(emotion_mode),
            "timing_mode": selection_value(timing_mode),
            "do_sample": toggle_value(do_sample),
            "top_p": number_value(4, 5),
            "top_k": number_value(30),
            "temperature": number_value(4, 5),
            "length_penalty": number_value(0),
            "num_beams": number_value(3),
            "repetition_penalty": number_value(10, 1),
            "max_mel_tokens": number_value(1500),
            "max_text_tokens_per_segment": number_value(120),
            "intersegment_silence": number_value(1, 5),
            "glossary": TableValue(kind="table", rows=[]),
        }
        if model == "IndexTeam/IndexTTS-2.5":
            inputs["language"] = selection_value("en")
        if timing_mode == "speed_factor":
            inputs["speed_factor"] = number_value(1)
        else:
            inputs["duration"] = number_value(3)
        if emotion_mode == "audio":
            inputs["emotion_audio"] = audio_value()
            inputs["emotion_strength"] = number_value(1)
        elif emotion_mode == "text":
            inputs["emotion_text"] = text_value("calm")
            inputs["emotion_strength"] = number_value(1, 2)
            inputs["randomize_emotion"] = toggle_value(False)
        elif emotion_mode == "factors":
            for name in (
                "happy",
                "angry",
                "sad",
                "afraid",
                "disgusted",
                "melancholic",
                "surprised",
                "calm",
            ):
                inputs[f"emotion_{name}"] = number_value(0)
            inputs["emotion_strength"] = number_value(1)
            inputs["randomize_emotion"] = toggle_value(True)
        if do_sample:
            inputs["typical_sampling"] = toggle_value(True)
            inputs["typical_mass"] = number_value(9, 10)
        return inputs

    def test_custom_voice_request_parses(self) -> None:
        parsed = parse_request(
            request(
                "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                **self.qwen_custom_inputs(),
            )
        )
        self.assertIsInstance(parsed, QwenCustomVoiceRequest)
        self.assertEqual(parsed.instruction, None)
        self.assertEqual(parsed.language, "English")

    def test_custom_voice_instruction_is_only_active_for_large_model(self) -> None:
        parsed = parse_request(
            request(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                **self.qwen_custom_inputs(instruction="gentle"),
            )
        )
        self.assertIsInstance(parsed, QwenCustomVoiceRequest)
        self.assertEqual(parsed.instruction, "gentle")
        with self.assertRaisesRegex(ValueError, "Unknown or inactive inputs"):
            parse_request(
                request(
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    **self.qwen_custom_inputs(instruction="unsupported"),
                )
            )

    def test_voice_clone_strips_blank_reference_transcript(self) -> None:
        parsed = parse_request(
            request(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                **self.qwen_clone_inputs("  "),
            )
        )
        self.assertIsInstance(parsed, QwenVoiceCloneRequest)
        self.assertTrue(parsed.audio_only)
        self.assertIsNone(parsed.reference_text)

    def test_voice_clone_keeps_reference_transcript(self) -> None:
        parsed = parse_request(
            request(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                **self.qwen_clone_inputs(" reference words "),
            )
        )
        self.assertIsInstance(parsed, QwenVoiceCloneRequest)
        self.assertFalse(parsed.audio_only)
        self.assertEqual(parsed.reference_text, "reference words")

    def test_voice_design_request_parses(self) -> None:
        parsed = parse_request(
            request(
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                precision=selection_value("float32"),
                text=text_value("Design a voice"),
                language=selection_value("English"),
                instruction=text_value("warm and clear"),
            )
        )
        self.assertIsInstance(parsed, QwenVoiceDesignRequest)
        self.assertEqual(parsed.instruction, "warm and clear")

    def test_qwen_parser_rejects_invalid_selection_and_missing_input(self) -> None:
        inputs = self.qwen_custom_inputs()
        inputs["language"] = selection_value("Klingon")
        with self.assertRaisesRegex(ValueError, "invalid selection"):
            parse_request(
                request("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", **inputs)
            )
        inputs = self.qwen_custom_inputs()
        del inputs["speaker"]
        with self.assertRaisesRegex(ValueError, "Missing inputs"):
            parse_request(
                request("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", **inputs)
            )

    def test_index_speaker_speed_request_parses_without_sampling_controls(self) -> None:
        parsed = parse_request(
            request("IndexTeam/IndexTTS-2", **self.index_inputs())
        )
        self.assertIsInstance(parsed, IndexTts2Request)
        self.assertIsInstance(parsed.timing, SpeedTiming)
        self.assertEqual(parsed.timing.factor, Fraction(1))
        self.assertEqual(parsed.emotion.mode, "speaker")
        self.assertFalse(parsed.sampling.do_sample)
        self.assertFalse(parsed.sampling.typical_sampling)
        self.assertEqual(parsed.glossary, [])

    def test_index_25_audio_duration_and_typical_sampling_parse(self) -> None:
        parsed = parse_request(
            request(
                "IndexTeam/IndexTTS-2.5",
                **self.index_inputs(
                    model="IndexTeam/IndexTTS-2.5",
                    emotion_mode="audio",
                    timing_mode="duration",
                    do_sample=True,
                ),
            )
        )
        self.assertIsInstance(parsed, IndexTts2Request)
        self.assertEqual(parsed.language, "en")
        self.assertIsInstance(parsed.timing, DurationTiming)
        self.assertEqual(parsed.timing.seconds, Fraction(3))
        self.assertEqual(parsed.emotion.mode, "audio")
        self.assertEqual(parsed.sampling.typical_mass, 0.9)

    def test_index_emotion_factor_and_text_branches_parse(self) -> None:
        factor_request = request(
            "IndexTeam/IndexTTS-2",
            **self.index_inputs(emotion_mode="factors"),
        )
        factor_parsed = parse_request(factor_request)
        self.assertIsInstance(factor_parsed.emotion, FactorEmotion)
        self.assertEqual(
            factor_parsed.emotion.factors.values_in_model_order(),
            [0.0] * 8,
        )

        text_parsed = parse_request(
            request(
                "IndexTeam/IndexTTS-2",
                **self.index_inputs(emotion_mode="text"),
            )
        )
        self.assertIsInstance(text_parsed.emotion, TextEmotion)
        self.assertEqual(text_parsed.emotion.text, "calm")
        self.assertFalse(text_parsed.emotion.randomize)

    def test_index_model_constraints_and_glossary(self) -> None:
        inputs = self.index_inputs()
        inputs["glossary"] = TableValue(
            kind="table",
            rows=[{"term": "GPU", "english": "gee pee you"}],
        )
        parsed = parse_request(request("IndexTeam/IndexTTS-2", **inputs))
        self.assertEqual(parsed.glossary[0].term, "GPU")
        self.assertEqual(parsed.glossary[0].english, "gee pee you")

        valid_index_25 = parse_request(
            request(
                "IndexTeam/IndexTTS-2.5",
                **self.index_inputs(model="IndexTeam/IndexTTS-2.5"),
            )
        )
        with self.assertRaisesRegex(ValidationError, "requires a language"):
            IndexTts2Request(
                model="IndexTeam/IndexTTS-2.5",
                text=valid_index_25.text,
                language=None,
                voice=valid_index_25.voice,
                emotion=valid_index_25.emotion,
                timing=valid_index_25.timing,
                sampling=valid_index_25.sampling,
                max_text_tokens_per_segment=valid_index_25.max_text_tokens_per_segment,
                intersegment_silence=valid_index_25.intersegment_silence,
                glossary=valid_index_25.glossary,
            )

        with self.assertRaisesRegex(ValidationError, "does not accept a language"):
            IndexTts2Request(
                model="IndexTeam/IndexTTS-2",
                text=valid_index_25.text,
                language="en",
                voice=valid_index_25.voice,
                emotion=valid_index_25.emotion,
                timing=valid_index_25.timing,
                sampling=valid_index_25.sampling,
                max_text_tokens_per_segment=valid_index_25.max_text_tokens_per_segment,
                intersegment_silence=valid_index_25.intersegment_silence,
                glossary=valid_index_25.glossary,
            )

    def test_validate_inputs_checks_visibility_table_and_value_type(self) -> None:
        inputs = self.index_inputs(emotion_mode="speaker")
        inputs["emotion_audio"] = audio_value()
        with self.assertRaisesRegex(ValueError, "Unknown or inactive inputs"):
            validate_inputs(request("IndexTeam/IndexTTS-2", **inputs))

        inputs = self.qwen_custom_inputs()
        inputs["text"] = selection_value("wrong kind")
        with self.assertRaisesRegex(ValueError, "wrong value type"):
            validate_inputs(
                request("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", **inputs)
            )

        inputs = self.qwen_custom_inputs()
        inputs["text"] = text_value("x" * 4097)
        with self.assertRaisesRegex(ValueError, "too long"):
            validate_inputs(
                request("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", **inputs)
            )

    def test_table_validation_rejects_unknown_required_and_long_cells(self) -> None:
        inputs = self.index_inputs()
        inputs["glossary"] = TableValue(
            kind="table", rows=[{"term": "x", "unknown": "y"}]
        )
        with self.assertRaisesRegex(ValueError, "unknown columns"):
            validate_inputs(request("IndexTeam/IndexTTS-2", **inputs))

        inputs = self.index_inputs()
        inputs["glossary"] = TableValue(kind="table", rows=[{"term": ""}])
        with self.assertRaisesRegex(ValueError, "column term is required"):
            validate_inputs(request("IndexTeam/IndexTTS-2", **inputs))

        inputs = self.index_inputs()
        inputs["glossary"] = TableValue(
            kind="table", rows=[{"term": "x" * 257}]
        )
        with self.assertRaisesRegex(ValueError, "column term is too long"):
            validate_inputs(request("IndexTeam/IndexTTS-2", **inputs))

    def test_typed_accessors_and_request_text(self) -> None:
        parsed_request = request(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            **self.qwen_custom_inputs(),
        )
        self.assertEqual(text(parsed_request, "text"), "Hello")
        self.assertEqual(selection(parsed_request, "precision"), "auto")
        self.assertEqual(request_precision(parsed_request), "auto")
        self.assertEqual(request_text(parsed_request), "Hello")
        self.assertEqual(request_text(request(parsed_request.model)), "")
        with self.assertRaisesRegex(ValueError, "wrong value type"):
            text(parsed_request, "precision")
        with self.assertRaisesRegex(ValueError, "wrong value type"):
            audio(parsed_request, "text")
        with self.assertRaisesRegex(ValueError, "wrong value type"):
            toggle(parsed_request, "text")

    def test_integer_rejects_fractional_number(self) -> None:
        inputs = self.index_inputs()
        inputs["top_k"] = number_value(3, 2)
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            parse_request(request("IndexTeam/IndexTTS-2", **inputs))

    def test_timing_models_reject_nonpositive_values(self) -> None:
        with self.assertRaisesRegex(ValidationError, "Speed factor must be positive"):
            SpeedTiming(factor=Fraction(0))
        with self.assertRaisesRegex(ValidationError, "Duration must be positive"):
            DurationTiming(seconds=Fraction(0))


if __name__ == "__main__":
    import unittest

    unittest.main()
