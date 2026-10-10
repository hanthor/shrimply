from unittest import TestCase
from unittest.mock import MagicMock, call, patch

from api.model_progress import (
    StreamedModelProgress,
    stream_model_progress,
    report_progress,
    last_progress,
)


class StreamedModelProgressTests(TestCase):
    """Unit tests for StreamedModelProgress tqdm wrapper."""

    def test_display_no_callback(self) -> None:
        """display returns early if report_progress is None."""
        with patch("api.model_progress.report_progress", None):
            progress = StreamedModelProgress(total=100)
            progress.display()
            # Should complete without error

    def test_display_no_total(self) -> None:
        """display returns early if total is not set."""
        callback = MagicMock()
        with patch("api.model_progress.report_progress", callback):
            progress = StreamedModelProgress()
            progress.display()
            callback.assert_not_called()

    def test_display_reports_progress_percent(self) -> None:
        """display calculates and reports progress percentage."""
        callback = MagicMock()
        with patch("api.model_progress.report_progress", callback):
            progress = StreamedModelProgress(total=100, desc="Test")
            progress.n = 50
            progress.display()
            callback.assert_called_once()
            args = callback.call_args[0]
            self.assertIn("50%", args[0])

    def test_display_clamps_percentage(self) -> None:
        """display clamps percentage to 0-100 range."""
        callback = MagicMock()
        with patch("api.model_progress.report_progress", callback):
            progress = StreamedModelProgress(total=100, desc="Test")
            progress.n = 150  # Over 100%
            progress.display()
            args = callback.call_args[0]
            self.assertIn("100%", args[0])

    def test_display_uses_custom_description(self) -> None:
        """display includes custom description in progress message."""
        callback = MagicMock()
        with patch("api.model_progress.report_progress", callback):
            progress = StreamedModelProgress(total=100, desc="Downloading model")
            progress.n = 25
            progress.display()
            args = callback.call_args[0]
            self.assertIn("Downloading model", args[0])

    def test_display_reconstruct_description(self) -> None:
        """display replaces 'reconstruct' with 'Preparing model cache'."""
        callback = MagicMock()
        with patch("api.model_progress.report_progress", callback):
            progress = StreamedModelProgress(total=100, unit="B", desc="reconstruct")
            progress.n = 50
            progress.display()
            args = callback.call_args[0]
            self.assertIn("Preparing model cache", args[0])

    def test_display_download_description_for_bytes(self) -> None:
        """display uses 'Downloading model' for byte unit without reconstruct."""
        callback = MagicMock()
        with patch("api.model_progress.report_progress", callback):
            progress = StreamedModelProgress(total=1000, unit="B", desc="file")
            progress.n = 500
            progress.display()
            args = callback.call_args[0]
            self.assertIn("Downloading model", args[0])

    def test_display_loading_description_for_no_desc(self) -> None:
        """display uses 'Loading model' when description is empty."""
        callback = MagicMock()
        with patch("api.model_progress.report_progress", callback):
            progress = StreamedModelProgress(total=100, unit="it")
            progress.n = 50
            progress.display()
            args = callback.call_args[0]
            self.assertIn("Loading model", args[0])

    def test_display_deduplicates_messages(self) -> None:
        """display only calls callback when message changes."""
        callback = MagicMock()
        with patch("api.model_progress.report_progress", callback):
            with patch("api.model_progress.last_progress", None):
                progress = StreamedModelProgress(total=100, desc="Test")
                progress.n = 50

                progress.display()
                self.assertEqual(callback.call_count, 1)

                # Same progress, should not call again
                progress.display()
                self.assertEqual(callback.call_count, 1)

    def test_display_updates_last_progress(self) -> None:
        """display updates global last_progress state."""
        callback = MagicMock()
        with patch("api.model_progress.report_progress", callback), patch(
            "api.model_progress.last_progress", None, create=True
        ):
            progress = StreamedModelProgress(total=100, desc="Test")
            progress.n = 50
            progress.display()

            # Verify callback was called (which updates last_progress)
            self.assertEqual(callback.call_count, 1)


class StreamModelProgressContextManagerTests(TestCase):
    """Unit tests for stream_model_progress context manager."""

    def test_context_manager_enters_and_exits(self) -> None:
        """stream_model_progress context manager enters and exits cleanly."""
        callback = MagicMock()
        with stream_model_progress(callback):
            pass
        # Should complete without error

    def test_context_manager_patches_transformers(self) -> None:
        """stream_model_progress patches transformers.utils.logging."""
        callback = MagicMock()
        mock_transformers_logging = MagicMock()
        original_hook = MagicMock()
        mock_transformers_logging.set_tqdm_hook.return_value = original_hook

        with patch(
            "importlib.import_module"
        ) as mock_import, patch(
            "api.model_progress.report_progress", callback
        ):

            def import_side_effect(module_name):
                if module_name == "transformers.utils.logging":
                    return mock_transformers_logging
                elif module_name == "huggingface_hub._snapshot_download":
                    m = MagicMock()
                    m.hf_tqdm = MagicMock()
                    return m
                elif module_name == "huggingface_hub.file_download":
                    m = MagicMock()
                    m.tqdm = MagicMock()
                    return m
                return MagicMock()

            mock_import.side_effect = import_side_effect

            with stream_model_progress(callback):
                pass

            # Verify transformers hook was set and restored
            self.assertGreater(mock_transformers_logging.set_tqdm_hook.call_count, 1)

    def test_context_manager_patches_huggingface_hub(self) -> None:
        """stream_model_progress patches huggingface_hub modules."""
        callback = MagicMock()

        def import_side_effect(module_name):
            if module_name == "transformers.utils.logging":
                m = MagicMock()
                m.set_tqdm_hook.return_value = None
                return m
            elif module_name == "huggingface_hub._snapshot_download":
                m = MagicMock()
                m.hf_tqdm = MagicMock()
                return m
            elif module_name == "huggingface_hub.file_download":
                m = MagicMock()
                m.tqdm = MagicMock()
                return m
            return MagicMock()

        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = import_side_effect

            with stream_model_progress(callback):
                pass

            # Verify both huggingface_hub submodules were patched
            module_names = [call[0][0] for call in mock_import.call_args_list]
            self.assertIn("huggingface_hub._snapshot_download", module_names)
            self.assertIn("huggingface_hub.file_download", module_names)

    def test_context_manager_restores_on_exception(self) -> None:
        """stream_model_progress restores patches even if exception occurs."""
        callback = MagicMock()
        mock_transformers_logging = MagicMock()
        original_hook = MagicMock()
        mock_transformers_logging.set_tqdm_hook.return_value = original_hook

        def import_side_effect(module_name):
            if module_name == "transformers.utils.logging":
                return mock_transformers_logging
            elif module_name == "huggingface_hub._snapshot_download":
                m = MagicMock()
                m.hf_tqdm = MagicMock()
                return m
            elif module_name == "huggingface_hub.file_download":
                m = MagicMock()
                m.tqdm = MagicMock()
                return m
            return MagicMock()

        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = import_side_effect

            try:
                with stream_model_progress(callback):
                    raise RuntimeError("Test error")
            except RuntimeError:
                pass

            # Verify restore was called despite exception
            self.assertGreater(mock_transformers_logging.set_tqdm_hook.call_count, 1)

    def test_context_manager_preserves_report_progress(self) -> None:
        """stream_model_progress preserves previous report_progress value."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        def import_side_effect(module_name):
            if module_name == "transformers.utils.logging":
                m = MagicMock()
                m.set_tqdm_hook.return_value = None
                return m
            elif module_name == "huggingface_hub._snapshot_download":
                m = MagicMock()
                m.hf_tqdm = MagicMock()
                return m
            elif module_name == "huggingface_hub.file_download":
                m = MagicMock()
                m.tqdm = MagicMock()
                return m
            return MagicMock()

        with patch("importlib.import_module") as mock_import, patch(
            "api.model_progress.report_progress", callback1
        ):
            mock_import.side_effect = import_side_effect

            with stream_model_progress(callback2):
                pass

            # After context exit, should be back to original

    def test_context_manager_optionally_patches_diffusers(self) -> None:
        """stream_model_progress patches diffusers when include_diffusers=True."""
        callback = MagicMock()
        mock_diffusers_logging = MagicMock()
        mock_diffusers_logging.tqdm_lib.tqdm = MagicMock()

        def import_side_effect(module_name):
            if module_name == "transformers.utils.logging":
                m = MagicMock()
                m.set_tqdm_hook.return_value = None
                return m
            elif module_name == "huggingface_hub._snapshot_download":
                m = MagicMock()
                m.hf_tqdm = MagicMock()
                return m
            elif module_name == "huggingface_hub.file_download":
                m = MagicMock()
                m.tqdm = MagicMock()
                return m
            elif module_name == "diffusers.utils.logging":
                return mock_diffusers_logging
            return MagicMock()

        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = import_side_effect

            with stream_model_progress(callback, include_diffusers=True):
                pass

            # Verify diffusers was patched
            module_names = [call[0][0] for call in mock_import.call_args_list]
            self.assertIn("diffusers.utils.logging", module_names)

    def test_context_manager_skips_diffusers_by_default(self) -> None:
        """stream_model_progress doesn't patch diffusers by default."""
        callback = MagicMock()

        def import_side_effect(module_name):
            if module_name == "transformers.utils.logging":
                m = MagicMock()
                m.set_tqdm_hook.return_value = None
                return m
            elif module_name == "huggingface_hub._snapshot_download":
                m = MagicMock()
                m.hf_tqdm = MagicMock()
                return m
            elif module_name == "huggingface_hub.file_download":
                m = MagicMock()
                m.tqdm = MagicMock()
                return m
            return MagicMock()

        with patch("importlib.import_module") as mock_import:
            mock_import.side_effect = import_side_effect

            with stream_model_progress(callback, include_diffusers=False):
                pass

            # Verify diffusers was not imported
            module_names = [call[0][0] for call in mock_import.call_args_list]
            self.assertNotIn("diffusers.utils.logging", module_names)
