import importlib.util
import sys
import types

if importlib.util.find_spec("torch") is None:
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch_stub

from unittest import TestCase

from pydantic import ValidationError

from api.tracking_3d.protocol import (
    AnalysisRequest,
    CameraEvent,
    ErrorEvent,
    ProgressEvent,
    ResultEvent,
    decode_event,
    encode_event,
)


class AnalysisRequestTests(TestCase):
    def test_colmap_requires_backend_settings(self) -> None:
        request = AnalysisRequest(
            version=1,
            model="colmap/colmap",
            frame_count=2,
            quality="high",
            camera_model="pinhole",
        )

        self.assertEqual(request.model, "colmap/colmap")
        self.assertEqual(request.quality, "high")
        self.assertEqual(request.camera_model, "pinhole")

    def test_colmap_rejects_missing_backend_settings(self) -> None:
        with self.assertRaisesRegex(
            ValidationError, "COLMAP quality and camera model are required"
        ):
            AnalysisRequest(version=1, model="colmap/colmap", frame_count=2)

    def test_vggt_rejects_colmap_backend_settings(self) -> None:
        with self.assertRaisesRegex(
            ValidationError, "COLMAP settings cannot be used with this backend"
        ):
            AnalysisRequest(
                version=1,
                model="MIT-SPARK/VGGT-SLAM",
                frame_count=2,
                quality="medium",
                camera_model="simple_radial",
            )

    def test_request_rejects_single_frame(self) -> None:
        with self.assertRaises(ValidationError):
            AnalysisRequest(
                version=1,
                model="MIT-SPARK/VGGT-SLAM",
                frame_count=1,
            )


class CameraEventTests(TestCase):
    def test_equirectangular_camera_has_no_focal_length(self) -> None:
        event = CameraEvent(
            frame_index=0,
            camera_from_world_rotation=[1.0, 0.0, 0.0, 0.0],
            camera_from_world_translation=[0.0, 0.0, 0.0],
            projection="equirectangular",
            image_width=1920,
            image_height=960,
            focal_y=None,
        )

        self.assertIsNone(event.focal_y)

    def test_perspective_camera_requires_positive_focal_length(self) -> None:
        with self.assertRaisesRegex(
            ValidationError, "require a focal length"
        ):
            CameraEvent(
                frame_index=0,
                camera_from_world_rotation=[1.0, 0.0, 0.0, 0.0],
                camera_from_world_translation=[0.0, 0.0, 0.0],
                projection="perspective",
                image_width=1920,
                image_height=1080,
                focal_y=None,
            )

        with self.assertRaises(ValidationError):
            CameraEvent(
                frame_index=0,
                camera_from_world_rotation=[1.0, 0.0, 0.0, 0.0],
                camera_from_world_translation=[0.0, 0.0, 0.0],
                projection="fisheye",
                image_width=1920,
                image_height=1080,
                focal_y=0,
            )

    def test_equirectangular_camera_rejects_focal_length(self) -> None:
        with self.assertRaisesRegex(
            ValidationError, "cannot have a focal length"
        ):
            CameraEvent(
                frame_index=0,
                camera_from_world_rotation=[1.0, 0.0, 0.0, 0.0],
                camera_from_world_translation=[0.0, 0.0, 0.0],
                projection="equirectangular",
                image_width=1920,
                image_height=960,
                focal_y=960.0,
            )


class EventCodecTests(TestCase):
    def test_encode_decode_preserves_worker_event(self) -> None:
        events = (
            ProgressEvent(
                message="Analyzing frames",
                completed_frames=1,
                total_frames=2,
            ),
            CameraEvent(
                frame_index=1,
                camera_from_world_rotation=[1.0, 0.0, 0.0, 0.0],
                camera_from_world_translation=[0.0, 0.0, 0.0],
                projection="perspective",
                image_width=1920,
                image_height=1080,
                focal_y=960.0,
            ),
            ResultEvent(camera_count=2),
            ErrorEvent(code="analysis_failed", message="model failed"),
        )

        for event in events:
            with self.subTest(kind=event.kind):
                self.assertEqual(decode_event(encode_event(event)), event)

    def test_decode_rejects_unknown_event_kind(self) -> None:
        import msgspec

        payload = msgspec.msgpack.encode(
            {"kind": "unknown", "message": "unsupported"}
        )

        with self.assertRaises(ValidationError):
            decode_event(payload)
