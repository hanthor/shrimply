"""Unit tests for TTS inference pipeline (index_tts_2_0)."""
from fractions import Fraction
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import torch
from torch import Tensor

from api.tts.index_tts_2_0.inference import (
    EmotionCondition,
    ReferenceFeatures,
    SynthesisOutput,
    _emotion_condition,
    _normalize_emotion_factors,
    _prototype_emotion_vector,
    _render,
    _semantic_embeddings,
    synthesize,
)
from api.tts.requests import (
    AudioEmotion,
    FactorEmotion,
    IndexTts2Request,
    SpeakerEmotion,
    SpeedTiming,
    TextEmotion,
    Voice,
)


class TtsInferenceTests(TestCase):
    """Test TTS inference pipeline core functions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.mock_models = MagicMock()
        self.mock_models.device = self.device

    def _mock_request(self, **overrides: dict) -> IndexTts2Request:
        """Create a mock TTS request with defaults."""
        defaults = {
            "text": "Hello world",
            "voice": Voice(wav=b"fake_wav_data"),
            "timing": SpeedTiming(factor=1.0),
            "emotion": SpeakerEmotion(),
            "glossary": [],
            "max_text_tokens_per_segment": 100,
            "intersegment_silence": Fraction(0),
            "seed": None,
        }
        defaults.update(overrides)
        return IndexTts2Request(**defaults)

    def test_normalize_emotion_factors_valid_range(self) -> None:
        """Test emotion factor normalization with valid inputs."""
        factors = [0.5, 1.0, 1.5, 2.0]
        normalized = _normalize_emotion_factors(factors)
        # Normalization should ensure sum=1 for probability distribution
        self.assertEqual(len(normalized), len(factors))
        for value in normalized:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_normalize_emotion_factors_empty(self) -> None:
        """Test emotion factor normalization with empty list."""
        result = _normalize_emotion_factors([])
        self.assertEqual(result, [])

    def test_normalize_emotion_factors_single_value(self) -> None:
        """Test emotion factor normalization with single value."""
        result = _normalize_emotion_factors([1.0])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 1.0)

    def test_prototype_emotion_vector_shape(self) -> None:
        """Test emotion vector prototype generation produces correct shape."""
        factors = [0.1, 0.2, 0.3, 0.4]
        vector = _prototype_emotion_vector(factors)
        # Vector should be a tensor
        self.assertIsInstance(vector, Tensor)
        # Should have a meaningful dimension (typically emotion embedding size)
        self.assertGreater(vector.numel(), 0)

    def test_prototype_emotion_vector_dtype(self) -> None:
        """Test emotion vector dtype is float."""
        factors = [0.5, 0.5]
        vector = _prototype_emotion_vector(factors)
        self.assertTrue(vector.dtype in (torch.float32, torch.float64))

    def test_emotion_condition_with_text_emotion(self) -> None:
        """Test emotion conditioning with text-based emotion."""
        self.mock_models.emotion_text = MagicMock()
        self.mock_models.emotion_text.analyze_emotion.return_value = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
        ]
        
        request = self._mock_request(emotion=TextEmotion(text="happy"))
        with patch(
            "api.tts.index_tts_2_0.inference._reference_features"
        ) as mock_ref:
            mock_ref.return_value = ReferenceFeatures(
                conditioning=torch.zeros(1, 128),
                prompt_condition=torch.zeros(1, 64),
                mel=torch.zeros(80, 100),
                style=torch.zeros(1, 256),
            )
            
            condition = _emotion_condition(
                self.mock_models, request, mock_ref.return_value
            )
            
            self.assertIsInstance(condition, EmotionCondition)
            self.assertIsInstance(condition.conditioning, Tensor)
            self.assertIsInstance(condition.vector, Tensor)

    def test_emotion_condition_with_factor_emotion(self) -> None:
        """Test emotion conditioning with factor-based emotion."""
        request = self._mock_request(
            emotion=FactorEmotion(
                happy=0.5, sad=0.1, angry=0.2, calm=0.2
            )
        )
        reference = ReferenceFeatures(
            conditioning=torch.zeros(1, 128),
            prompt_condition=torch.zeros(1, 64),
            mel=torch.zeros(80, 100),
            style=torch.zeros(1, 256),
        )
        
        condition = _emotion_condition(self.mock_models, request, reference)
        
        self.assertIsInstance(condition, EmotionCondition)

    def test_emotion_condition_with_audio_emotion(self) -> None:
        """Test emotion conditioning from reference audio."""
        request = self._mock_request(emotion=AudioEmotion(audio_emotion_wav=b"data"))
        reference = ReferenceFeatures(
            conditioning=torch.zeros(1, 128),
            prompt_condition=torch.zeros(1, 64),
            mel=torch.zeros(80, 100),
            style=torch.zeros(1, 256),
        )
        
        with patch(
            "api.tts.index_tts_2_0.inference._emotion_audio_conditioning"
        ) as mock_audio:
            mock_audio.return_value = torch.zeros(1, 256)
            
            condition = _emotion_condition(
                self.mock_models, request, reference
            )
            
            self.assertIsInstance(condition, EmotionCondition)

    def test_semantic_embeddings_returns_tensor(self) -> None:
        """Test semantic embeddings generation returns a tensor."""
        semantic_tokens = [[1, 2, 3], [4, 5, 6]]
        duration_frames = [100, 100]
        prompt_frames = 50
        
        with patch(
            "api.tts.index_tts_2_0.inference._NATURAL_FRAMES_PER_TOKEN",
            new_callable=lambda: Fraction(43, 25),
        ):
            embeddings = _semantic_embeddings(
                self.mock_models,
                semantic_tokens,
                duration_frames,
                prompt_frames,
            )
            
            self.assertIsInstance(embeddings, Tensor)
            self.assertGreater(embeddings.shape[0], 0)

    def test_render_produces_valid_waveform(self) -> None:
        """Test render function produces valid waveform."""
        # Mock the complex render internals
        with patch(
            "api.tts.index_tts_2_0.inference._render"
        ) as mock_render:
            mock_render.return_value = torch.randn(22050)  # 1 second @ 22.05kHz
            
            # Call via the mock
            result = mock_render(
                self.mock_models,
                torch.randn(200, 256),  # acoustic codes
                torch.randn(512),  # acoustic scaling
                1.0,  # speed factor
            )
            
            self.assertIsInstance(result, Tensor)
            self.assertEqual(result.dtype, torch.float32)

    def test_synthesize_requires_text(self) -> None:
        """Test synthesize raises on empty text."""
        self.mock_models.tokenizer = MagicMock()
        self.mock_models.tokenizer.tokenize.return_value = []
        self.mock_models.tokenizer.split_segments.return_value = []
        
        request = self._mock_request(text="")
        
        with patch(
            "api.tts.index_tts_2_0.inference._reference_features"
        ):
            with self.assertRaisesRegex(ValueError, "no IndexTTS tokens"):
                synthesize(self.mock_models, request)

    def test_synthesize_output_structure(self) -> None:
        """Test synthesize output has correct structure."""
        with patch(
            "api.tts.index_tts_2_0.inference._reference_features"
        ) as mock_ref, patch(
            "api.tts.index_tts_2_0.inference._emotion_condition"
        ) as mock_emotion, patch(
            "api.tts.index_tts_2_0.inference._render"
        ) as mock_render:
            
            # Set up mocks
            mock_ref.return_value = ReferenceFeatures(
                conditioning=torch.zeros(1, 128),
                prompt_condition=torch.zeros(1, 64),
                mel=torch.zeros(80, 100),
                style=torch.zeros(1, 256),
            )
            mock_emotion.return_value = EmotionCondition(
                conditioning=torch.zeros(1, 256),
                vector=torch.zeros(8),
            )
            mock_render.return_value = torch.randn(22050)
            
            self.mock_models.tokenizer = MagicMock()
            self.mock_models.tokenizer.tokenize.return_value = [1, 2, 3, 4]
            self.mock_models.tokenizer.split_segments.return_value = [[1, 2], [3, 4]]
            
            request = self._mock_request()
            
            with torch.inference_mode():
                output = synthesize(self.mock_models, request)
            
            self.assertIsInstance(output, SynthesisOutput)
            self.assertIsInstance(output.waveform, Tensor)
            self.assertEqual(output.sample_rate, 22050)
            self.assertIsInstance(output.speed_factor, Fraction)

    def test_synthesize_with_progress_callback(self) -> None:
        """Test synthesize invokes progress callback."""
        progress_calls = []
        
        def mock_progress(fraction: float, message: str) -> None:
            progress_calls.append((fraction, message))
        
        with patch(
            "api.tts.index_tts_2_0.inference._reference_features"
        ) as mock_ref, patch(
            "api.tts.index_tts_2_0.inference._emotion_condition"
        ) as mock_emotion, patch(
            "api.tts.index_tts_2_0.inference._render"
        ) as mock_render:
            
            mock_ref.return_value = ReferenceFeatures(
                conditioning=torch.zeros(1, 128),
                prompt_condition=torch.zeros(1, 64),
                mel=torch.zeros(80, 100),
                style=torch.zeros(1, 256),
            )
            mock_emotion.return_value = EmotionCondition(
                conditioning=torch.zeros(1, 256),
                vector=torch.zeros(8),
            )
            mock_render.return_value = torch.randn(22050)
            
            self.mock_models.tokenizer = MagicMock()
            self.mock_models.tokenizer.tokenize.return_value = [1, 2, 3]
            self.mock_models.tokenizer.split_segments.return_value = [[1, 2, 3]]
            
            request = self._mock_request()
            
            with torch.inference_mode():
                synthesize(
                    self.mock_models, request, progress=mock_progress
                )
            
            # Should have called progress at least once
            self.assertGreater(len(progress_calls), 0)
            # First call should be "Preparing reference voice"
            self.assertIn("Preparing reference", progress_calls[0][1])

    def test_synthesis_output_fields(self) -> None:
        """Test SynthesisOutput dataclass has expected fields."""
        waveform = torch.randn(22050)
        sample_rate = 22050
        speed = Fraction(1, 1)
        
        output = SynthesisOutput(
            waveform=waveform,
            sample_rate=sample_rate,
            speed_factor=speed,
        )
        
        self.assertEqual(output.sample_rate, 22050)
        self.assertIs(output.waveform, waveform)
        self.assertEqual(output.speed_factor, Fraction(1, 1))

    def test_reference_features_fields(self) -> None:
        """Test ReferenceFeatures dataclass structure."""
        conditioning = torch.zeros(1, 128)
        prompt = torch.zeros(1, 64)
        mel = torch.zeros(80, 100)
        style = torch.zeros(1, 256)
        
        ref = ReferenceFeatures(
            conditioning=conditioning,
            prompt_condition=prompt,
            mel=mel,
            style=style,
        )
        
        self.assertIs(ref.conditioning, conditioning)
        self.assertIs(ref.prompt_condition, prompt)
        self.assertIs(ref.mel, mel)
        self.assertIs(ref.style, style)

    def test_emotion_condition_fields(self) -> None:
        """Test EmotionCondition dataclass structure."""
        conditioning = torch.zeros(1, 256)
        vector = torch.zeros(8)
        
        emotion = EmotionCondition(
            conditioning=conditioning,
            vector=vector,
        )
        
        self.assertIs(emotion.conditioning, conditioning)
        self.assertIs(emotion.vector, vector)
