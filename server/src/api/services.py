from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import Response
from pydantic import BaseModel

from api import gpu
from api.pneuma.catalog import models as pneuma_models
from api.pneuma.http import handle_request as handle_pneuma_request
from api.pneuma.protocol import ModelsResponse as PneumaModelsResponse
from api.sam2.http import handle_request as handle_sam2_request
from api.sam2.protocol import MODEL_IDS as SAM2_MODEL_IDS
from api.service import Service
from api.stt.http import handle_request as handle_stt_request
from api.stt.protocol import MODEL_IDS as STT_MODEL_IDS
from api.tracking_3d.http import handle_request as handle_tracking_3d_request
from api.tracking_3d.protocol import MODEL_IDS as TRACKING_3D_MODEL_IDS
from api.tts.catalog import MODELS as TTS_MODELS
from api.tts.http import handle_request as handle_tts_request
from api.tts.protocol import MODEL_IDS as TTS_MODEL_IDS
from api.tts.protocol import ModelsResponse as TtsModelsResponse
from api.video_generation.catalog import MODELS as VIDEO_GENERATION_MODELS
from api.video_generation.http import handle_request as handle_video_generation_request
from api.video_generation.protocol import MODEL_IDS as VIDEO_GENERATION_MODEL_IDS
from api.video_generation.protocol import (
    ModelsResponse as VideoGenerationModelsResponse,
)


type RequestHandler = Callable[[Request], Awaitable[Response]]
type ModelProvider = Callable[[], Iterable[str]]
type ModelAvailability = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    path: str
    response: Callable[[], BaseModel]


@dataclass(frozen=True, slots=True)
class ServiceDescriptor:
    service: Service
    capability: str
    path: str
    handler: RequestHandler
    models: ModelProvider
    available: ModelAvailability
    catalog: ModelCatalog | None = None


def _always_available(_model: str) -> bool:
    return True


def _pneuma_model_names() -> Iterable[str]:
    return (model.name for model in pneuma_models())


SERVICE_DESCRIPTORS = (
    ServiceDescriptor(
        service="stt",
        capability="stt",
        path="/transcriptions",
        handler=handle_stt_request,
        models=lambda: STT_MODEL_IDS,
        available=_always_available,
    ),
    ServiceDescriptor(
        service="tts",
        capability="tts",
        path="/speech",
        handler=handle_tts_request,
        models=lambda: TTS_MODEL_IDS,
        available=_always_available,
        catalog=ModelCatalog(
            path="/tts/models",
            response=lambda: TtsModelsResponse(models=TTS_MODELS),
        ),
    ),
    ServiceDescriptor(
        service="sam2",
        capability="sam2",
        path="/sam2/analyses",
        handler=handle_sam2_request,
        models=lambda: SAM2_MODEL_IDS,
        available=_always_available,
    ),
    ServiceDescriptor(
        service="pneuma",
        capability="pneuma",
        path="/pneuma/conversions",
        handler=handle_pneuma_request,
        models=_pneuma_model_names,
        available=_always_available,
        catalog=ModelCatalog(
            path="/pneuma/models",
            response=lambda: PneumaModelsResponse(models=pneuma_models()),
        ),
    ),
    ServiceDescriptor(
        service="video_generation",
        capability="video-generation",
        path="/video-generations",
        handler=handle_video_generation_request,
        models=lambda: VIDEO_GENERATION_MODEL_IDS,
        available=lambda _model: gpu.device.startswith("cuda:"),
        catalog=ModelCatalog(
            path="/video-generation/models",
            response=lambda: VideoGenerationModelsResponse(
                models=VIDEO_GENERATION_MODELS
            ),
        ),
    ),
    ServiceDescriptor(
        service="3dtracking",
        capability="3dtracking",
        path="/3dtracking/analyses",
        handler=handle_tracking_3d_request,
        models=lambda: TRACKING_3D_MODEL_IDS,
        available=lambda model: model != "MIT-SPARK/VGGT-SLAM"
        or gpu.device.startswith("cuda:"),
    ),
)


def service_capabilities() -> list[str]:
    return [
        f"{descriptor.capability}:{model}"
        for descriptor in SERVICE_DESCRIPTORS
        for model in descriptor.models()
        if descriptor.available(model)
    ]
