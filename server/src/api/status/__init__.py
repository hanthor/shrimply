from typing import Literal

import torch
from pydantic import BaseModel

from api import gpu, resource
from api.services import service_capabilities


class ProtocolStatus(BaseModel):
    major: int
    minor: int


class ServerVersion(BaseModel):
    version: str
    git_hash: str
    git_short_hash: str


class DeviceStatus(BaseModel):
    id: str
    name: str
    total_memory_bytes: int | None


class TorchStatus(BaseModel):
    version: str
    cuda_runtime: str | None
    cuda_available: bool
    devices: list[DeviceStatus]
    selected_device: str


class WorkerStatus(BaseModel):
    service: str
    model: str
    configuration: dict[str, str]
    state: str
    copies: int


class ComputeStatus(BaseModel):
    queued_jobs: int
    active_jobs: int
    reserved_ram_bytes: int
    reserved_vram_bytes: int
    workers: list[WorkerStatus]


class ServerStatus(BaseModel):
    protocol: ProtocolStatus
    server: ServerVersion
    status: Literal["ok", "degraded"]
    capabilities: list[str]
    torch: TorchStatus
    compute: ComputeStatus


def server_status(version: str, git_hash: str) -> ServerStatus:
    cuda_available = torch.cuda.is_available()
    devices = [DeviceStatus(id="cpu", name="CPU", total_memory_bytes=None)]
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                DeviceStatus(
                    id=f"cuda:{index}",
                    name=properties.name,
                    total_memory_bytes=properties.total_memory,
                )
            )
    return ServerStatus(
        protocol=ProtocolStatus(major=4, minor=0),
        server=ServerVersion(
            version=version,
            git_hash=git_hash,
            git_short_hash=git_hash[:8],
        ),
        status="ok",
        capabilities=service_capabilities(),
        torch=TorchStatus(
            version=str(torch.__version__),
            cuda_runtime=torch.version.cuda,
            cuda_available=cuda_available,
            devices=devices,
            selected_device=gpu.device,
        ),
        compute=ComputeStatus.model_validate(resource.scheduler.summary()),
    )
