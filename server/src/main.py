import logging
import subprocess
import tomllib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path

import env
from fastapi import Request
from fastapi.responses import Response
from gradio import Server

from api import (
    gpu,
    log_request,
    resource,
    respond,
    respond_error,
)
from api.services import ModelCatalog, SERVICE_DESCRIPTORS
from api.status import server_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("shrimply.server")

with (Path(__file__).parent.parent / "pyproject.toml").open("rb") as file:
    SERVER_VERSION = tomllib.load(file)["project"]["version"]


def read_git_hash() -> str:
    if env.SERVER_GIT_HASH:
        return env.SERVER_GIT_HASH
    try:
        return subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).parent.parent.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


SERVER_GIT_HASH = read_git_hash()


@asynccontextmanager
async def lifespan(_app: Server) -> AsyncGenerator[None]:
    try:
        yield
    finally:
        resource.shutdown_all()


def add_model_catalog_route(app: Server, catalog: ModelCatalog) -> None:
    def get_models() -> Response:
        return respond(HTTPStatus.OK, catalog.response())

    app.add_api_route(catalog.path, get_models, methods=["GET"])


def create_app() -> Server:
    app = Server(
        title="Shrimply Server",
        version=SERVER_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.get("/")
    def get_status() -> Response:
        return respond(HTTPStatus.OK, server_status(SERVER_VERSION, SERVER_GIT_HASH))


    @app.put("/compute/device")
    def select_compute_device(request: Request) -> Response:
        log_request(request, "compute-device selection")
        device_values = request.query_params.getlist("device")
        if len(device_values) != 1:
            return respond_error(
                HTTPStatus.BAD_REQUEST,
                "invalid_device",
                "Exactly one compute device is required",
            )
        try:
            gpu.select_device(device_values[0])
        except ValueError as exception:
            return respond_error(
                HTTPStatus.BAD_REQUEST, "invalid_device", str(exception)
            )
        except BlockingIOError as exception:
            return respond_error(HTTPStatus.CONFLICT, "compute_busy", str(exception))
        return respond(HTTPStatus.OK, server_status(SERVER_VERSION, SERVER_GIT_HASH))

    @app.put("/compute/jobs/{job_id}/heartbeat")
    def heartbeat_compute_job(job_id: str) -> Response:
        if not resource.heartbeat(job_id):
            return respond_error(
                HTTPStatus.NOT_FOUND, "job_not_found", "Compute job was not found"
            )
        return Response(status_code=HTTPStatus.NO_CONTENT)

    @app.delete("/compute/jobs/{job_id}")
    def cancel_compute_job(job_id: str) -> Response:
        resource.cancel(job_id)
        return Response(status_code=HTTPStatus.NO_CONTENT)

    for descriptor in SERVICE_DESCRIPTORS:
        if descriptor.catalog is not None:
            add_model_catalog_route(app, descriptor.catalog)
        app.add_api_route(descriptor.path, descriptor.handler, methods=["POST"])
    return app


def main(host: str, port: int, share: bool) -> None:
    logger.info("Shrimply server listening on http://%s:%d", host, port)
    create_app().launch(
        server_name=host,
        server_port=port,
        share=share,
    )


if __name__ == "__main__":
    main(env.SERVER_HOST, env.SERVER_PORT, env.SERVER_SHARE)
