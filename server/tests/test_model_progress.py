from types import ModuleType
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from api.model_progress import (
    StreamedModelProgress,
    stream_model_progress,
)


class TestStreamedModelProgress(TestCase):
    """Test StreamedModelProgress display logic."""

    def test_display_with_no_callback(self) -> None:
        """Test display does nothing when report_progress is None."""
        with patch("api.model_progress.report_progress", None):
            progress = StreamedModelProgress.__new__(StreamedModelProgress)
            progress.display()
            # Should not raise

    def test_display_with_no_total(self) -> None:
        """Test display does nothing when total is zero."""
        with patch("api.model_progress.report_progress", MagicMock()):
            progress = StreamedModelProgress.__new__(StreamedModelProgress)
            progress.total = 0
            progress.n = 0
            progress.desc = "test"
            progress.unit = ""
            progress.display()
            # Should not raise; report_progress should not be called

    def test_display_calculates_percent_correctly(self) -> None:
        """Test that display calculates percentage correctly."""
        mock_report = MagicMock()
        with patch("api.model_progress.report_progress", mock_report):
            progress = StreamedModelProgress.__new__(StreamedModelProgress)
            progress.total = 100
            progress.n = 50
            progress.desc = "Loading model"
            progress.unit = ""
            progress.display()
            mock_report.assert_called_once()
            call_arg = mock_report.call_args[0][0]
            assert "50%" in call_arg

    def test_display_clamps_percent_to_100(self) -> None:
        """Test that display clamps percent to 100%."""
        mock_report = MagicMock()
        with patch("api.model_progress.report_progress", mock_report):
            progress = StreamedModelProgress.__new__(StreamedModelProgress)
            progress.total = 100
            progress.n = 150  # > total
            progress.desc = "test"
            progress.unit = ""
            progress.display()
            mock_report.assert_called_once()
            call_arg = mock_report.call_args[0][0]
            assert "100%" in call_arg

    def test_display_custom_message_for_reconstruct(self) -> None:
        """Test that 'reconstruct' in description uses custom message."""
        mock_report = MagicMock()
        with patch("api.model_progress.report_progress", mock_report):
            progress = StreamedModelProgress.__new__(StreamedModelProgress)
            progress.total = 100
            progress.n = 50
            progress.desc = "Reconstruct model"
            progress.unit = "B"
            progress.display()
            call_arg = mock_report.call_args[0][0]
            assert "Preparing model cache" in call_arg

    def test_display_downloading_message_for_bytes(self) -> None:
        """Test that unit='B' (bytes) uses 'Downloading model' message."""
        mock_report = MagicMock()
        with patch("api.model_progress.report_progress", mock_report):
            progress = StreamedModelProgress.__new__(StreamedModelProgress)
            progress.total = 1000
            progress.n = 500
            progress.desc = "download"
            progress.unit = "B"
            progress.display()
            call_arg = mock_report.call_args[0][0]
            assert "Downloading model" in call_arg

    def test_display_loading_message_default(self) -> None:
        """Test that default empty description uses 'Loading model'."""
        mock_report = MagicMock()
        with patch("api.model_progress.report_progress", mock_report):
            progress = StreamedModelProgress.__new__(StreamedModelProgress)
            progress.total = 100
            progress.n = 50
            progress.desc = "  :  "  # stripped and rstrip(':') = empty
            progress.unit = "it"
            progress.display()
            call_arg = mock_report.call_args[0][0]
            assert "Loading model" in call_arg

    def test_display_caches_last_progress(self) -> None:
        """Test that repeated same messages are not reported."""
        mock_report = MagicMock()
        with patch("api.model_progress.report_progress", mock_report):
            with patch("api.model_progress.last_progress", None):
                progress = StreamedModelProgress.__new__(StreamedModelProgress)
                progress.total = 100
                progress.n = 50
                progress.desc = "test"
                progress.unit = ""
                progress.display()
                assert mock_report.call_count == 1
                # Second call with same state should not call report
                progress.display()
                # call_count should still be 1 (no new call)
                assert mock_report.call_count == 1


class TestStreamModelProgressContext(TestCase):
    """Test stream_model_progress context manager."""

    def test_context_manager_entry_exit(self) -> None:
        """Test that context manager enters and exits cleanly."""
        mock_report = MagicMock()
        with patch.dict("sys.modules", {
            "transformers.utils.logging": MagicMock(),
            "huggingface_hub._snapshot_download": MagicMock(),
            "huggingface_hub.file_download": MagicMock(),
        }):
            with stream_model_progress(mock_report):
                pass
            # Should not raise

    def test_context_manager_restores_previous_state(self) -> None:
        """Test that context manager restores library state after exit."""
        mock_report = MagicMock()
        
        # Create mock modules
        mock_transformers = MagicMock()
        mock_transformers.set_tqdm_hook = MagicMock(return_value="previous_hook")
        mock_snapshot = MagicMock()
        mock_snapshot.hf_tqdm = "original_tqdm"
        mock_file_download = MagicMock()
        mock_file_download.tqdm = "original_tqdm"
        
        with patch.dict("sys.modules", {
            "transformers.utils.logging": mock_transformers,
            "huggingface_hub._snapshot_download": mock_snapshot,
            "huggingface_hub.file_download": mock_file_download,
        }):
            with stream_model_progress(mock_report):
                pass
            
            # Verify transformers hook was restored
            mock_transformers.set_tqdm_hook.assert_called_with("previous_hook")

    def test_context_manager_restores_on_exception(self) -> None:
        """Test that library state is restored even if exception occurs."""
        mock_report = MagicMock()
        
        mock_transformers = MagicMock()
        mock_transformers.set_tqdm_hook = MagicMock(return_value="previous_hook")
        mock_snapshot = MagicMock()
        mock_snapshot.hf_tqdm = "original_tqdm"
        mock_file_download = MagicMock()
        mock_file_download.tqdm = "original_tqdm"
        
        with patch.dict("sys.modules", {
            "transformers.utils.logging": mock_transformers,
            "huggingface_hub._snapshot_download": mock_snapshot,
            "huggingface_hub.file_download": mock_file_download,
        }):
            try:
                with stream_model_progress(mock_report):
                    raise ValueError("test error")
            except ValueError:
                pass
            
            # State should still be restored even after exception
            mock_transformers.set_tqdm_hook.assert_called_with("previous_hook")

    def test_context_manager_sets_report_callback(self) -> None:
        """Test that report_progress is set in context."""
        mock_report = MagicMock()
        
        with patch.dict("sys.modules", {
            "transformers.utils.logging": MagicMock(),
            "huggingface_hub._snapshot_download": MagicMock(),
            "huggingface_hub.file_download": MagicMock(),
        }):
            with patch("api.model_progress.report_progress", None):
                # Inside context, report_progress should be set
                with stream_model_progress(mock_report):
                    import api.model_progress
                    assert api.model_progress.report_progress is mock_report

    def test_context_manager_include_diffusers_false_by_default(self) -> None:
        """Test that diffusers is not patched by default."""
        mock_report = MagicMock()
        
        with patch.dict("sys.modules", {
            "transformers.utils.logging": MagicMock(),
            "huggingface_hub._snapshot_download": MagicMock(),
            "huggingface_hub.file_download": MagicMock(),
        }):
            with patch("sys.modules.get", side_effect=lambda x: None):
                # Should not attempt to import diffusers
                with stream_model_progress(mock_report, include_diffusers=False):
                    pass

    def test_context_manager_include_diffusers_true(self) -> None:
        """Test that diffusers is patched when include_diffusers=True."""
        mock_report = MagicMock()
        
        mock_diffusers = MagicMock()
        mock_diffusers.tqdm_lib.tqdm = "original_diffusers_tqdm"
        
        mock_modules = {
            "transformers.utils.logging": MagicMock(),
            "huggingface_hub._snapshot_download": MagicMock(),
            "huggingface_hub.file_download": MagicMock(),
            "diffusers.utils.logging": mock_diffusers,
        }
        
        with patch.dict("sys.modules", mock_modules):
            with stream_model_progress(mock_report, include_diffusers=True):
                pass
            
            # Verify diffusers tqdm was restored
            assert mock_diffusers.tqdm_lib.tqdm == "original_diffusers_tqdm"
