from http import HTTPStatus
from unittest import TestCase, main
from unittest.mock import Mock

import msgspec
from fastapi.responses import Response

from api import validate_content_length

MESSAGE = "Request must be between 1 byte and 10 bytes"


def check(headers: dict[str, str]) -> int | Response:
    request = Mock()
    request.headers = headers
    return validate_content_length(request, 10, MESSAGE)


def error_of(response: Response) -> tuple[int, dict[str, str]]:
    body = msgspec.msgpack.decode(response.body)
    return response.status_code, body["error"]


class ValidateContentLengthTests(TestCase):
    def test_valid_length_is_returned(self) -> None:
        self.assertEqual(check({"content-length": "10"}), 10)

    def test_missing_header_is_length_required(self) -> None:
        response = check({})
        assert isinstance(response, Response)
        status, error = error_of(response)
        self.assertEqual(status, HTTPStatus.LENGTH_REQUIRED)
        self.assertEqual(error["code"], "content_length_required")

    def test_unparseable_header_is_length_required(self) -> None:
        response = check({"content-length": "abc"})
        assert isinstance(response, Response)
        self.assertEqual(response.status_code, HTTPStatus.LENGTH_REQUIRED)

    def test_out_of_range_is_bad_request_with_message(self) -> None:
        for value in ("0", "-1", "11"):
            response = check({"content-length": value})
            assert isinstance(response, Response)
            status, error = error_of(response)
            self.assertEqual(status, HTTPStatus.BAD_REQUEST)
            self.assertEqual(error["code"], "invalid_content_length")
            self.assertEqual(error["message"], MESSAGE)


if __name__ == "__main__":
    main()
