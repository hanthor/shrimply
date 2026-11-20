import asyncio
from collections.abc import Callable
from typing import Literal
from unittest import TestCase, main
from unittest.mock import Mock, patch

import msgspec
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api import stream_events


class StreamEvent(BaseModel):
    kind: Literal["result", "error"]
    message: str


async def collect(response: StreamingResponse) -> list[bytes]:
    return [chunk async for chunk in response.body_iterator]


def decode_frame(frame: bytes) -> dict[str, object]:
    payload_length = int.from_bytes(frame[:8], "little")
    payload = frame[8:]
    if payload_length != len(payload):
        raise AssertionError("stream frame length header does not match payload")
    return msgspec.msgpack.decode(payload)


class StreamEventsTests(TestCase):
    def test_successful_run_streams_event_and_finishes_job(self) -> None:
        job = Mock(id="job-id")
        cleanup = Mock()

        def run(send: Callable[[StreamEvent], None]) -> None:
            send(StreamEvent(kind="result", message="complete"))

        with (
            patch("api.resource.finish_job") as finish_job,
            patch("api.resource.scheduler.cancel") as cancel,
        ):
            response = stream_events(run, Mock(), "test-stream", job, cleanup)
            frames = asyncio.run(collect(response))

        self.assertEqual(len(frames), 1)
        self.assertEqual(
            decode_frame(frames[0]), {"kind": "result", "message": "complete"}
        )
        cleanup.assert_called_once_with()
        finish_job.assert_called_once_with(job)
        cancel.assert_called_once_with("job-id")
        job.check_cancelled.assert_called()

    def test_worker_failure_streams_failure_event_and_finishes_job(self) -> None:
        job = Mock(id="job-id")
        cleanup = Mock()
        failure = Mock(
            return_value=StreamEvent(kind="error", message="worker failed")
        )

        def run(_send: Callable[[StreamEvent], None]) -> None:
            raise RuntimeError("private failure")

        with (
            patch("api.resource.finish_job") as finish_job,
            patch("api.resource.scheduler.cancel") as cancel,
        ):
            response = stream_events(run, failure, "test-stream", job, cleanup)
            frames = asyncio.run(collect(response))

        self.assertEqual(len(frames), 1)
        self.assertEqual(
            decode_frame(frames[0]), {"kind": "error", "message": "worker failed"}
        )
        self.assertIsInstance(failure.call_args.args[0], RuntimeError)
        cleanup.assert_called_once_with()
        finish_job.assert_called_once_with(job)
        cancel.assert_called_once_with("job-id")


if __name__ == "__main__":
    main()
