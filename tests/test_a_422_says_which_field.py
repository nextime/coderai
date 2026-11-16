# CoderAI - OpenAI-compatible API server
# Copyright (C) 2026 Stefy Lanza <stefy@nexlab.net>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""A rejected body must say WHICH field it was rejected for.

A 422 is raised by the framework before any handler runs, so the only record
was the access line:

    POST /v1/loras/train/text HTTP/1.1" 422 Unprocessable Content
    POST /v1/loras/train/text HTTP/1.1" 422 Unprocessable Content
    POST /v1/loras/train/text HTTP/1.1" 200 OK

A client was rejected twice in thirty seconds and neither side could say what
was wrong: the client had the body and not the rule, this side had the rule
and logged neither.

The body is logged as SHAPE, not content, and that is deliberate rather than
squeamish: a training body on this route carries dataset rows, and a log
outlives the request it describes. Scalars go in full -- `pods: "8"` against
`pods: int` is exactly the mistake worth seeing -- and anything long, nested
or credential-shaped is reduced to a type, a size, or `<redacted>`.
"""

import pytest

from codai.api import app as app_mod


def test_a_credential_field_is_never_logged():
    for name in ("hf_token", "api_key", "Authorization", "PASSWORD",
                 "x-secret", "my_credential", "cookie", "bearer_tok"):
        assert app_mod._loggable(name, "sk-abcdef123456") == "<redacted>"


def test_a_scalar_is_logged_in_full_because_that_is_the_mistake():
    # The whole point: a client sending a string where an int belongs.
    assert app_mod._loggable("pods", "8") == "'8'"
    assert app_mod._loggable("pods", 8) == "8"
    assert app_mod._loggable("distributed", True) == "True"
    assert app_mod._loggable("rounds", None) == "None"


def test_a_long_string_is_a_length_not_a_copy():
    row = "Tribunale di Milano, sentenza n. 1234/2026 " * 40
    got = app_mod._loggable("prompt", row)
    assert "chars" in got
    assert len(got) < 200
    assert row not in got


def test_a_dataset_sized_list_is_a_count_not_its_rows():
    got = app_mod._loggable("examples", [{"q": "x"}] * 7311)
    assert got == "<list, 7311 item(s)>"


def test_a_nested_object_gives_its_keys_not_its_values():
    got = app_mod._loggable("field_map", {"prompt": "domanda",
                                          "answer": "risposta"})
    assert "prompt" in got and "answer" in got
    assert "domanda" not in got


def test_an_unknown_type_is_named_not_rendered():
    class Odd:
        def __repr__(self):
            raise AssertionError("repr must not be called on an odd value")
    assert app_mod._loggable("thing", Odd()) == "<Odd>"


# --------- the handler itself, through a real request ---------------------- #

def test_the_log_names_the_field_and_the_response_is_unchanged(capsys):
    """Driven through Starlette, so the handler is actually reached.

    A handler registered for the wrong exception class is silently never
    called -- which has happened in this file's neighbourhood before (a `500`
    key that never sees HTTPException(500)) -- so this asserts on output from
    a real rejected request rather than on the registration.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import BaseModel

    class Body(BaseModel):
        name: str
        pods: int

    probe = FastAPI()
    probe.add_exception_handler(
        __import__("fastapi.exceptions", fromlist=["x"]).RequestValidationError,
        app_mod.validation_error_says_which_field)

    @probe.post("/v1/loras/train/text")
    def _route(body: Body):
        return {"ok": True}

    client = TestClient(probe)
    r = client.post("/v1/loras/train/text",
                    json={"pods": "lots", "hf_token": "sk-secret-value",
                          "name": "adapter-x"})
    assert r.status_code == 422
    # ...and the answer still carries the detail a client needs.
    assert "detail" in r.json()

    out = capsys.readouterr().out
    assert "422 on POST /v1/loras/train/text" in out
    assert "pods" in out                      # the field that failed
    assert "body shape" in out
    assert "<redacted>" in out                # the token did not get logged
    assert "sk-secret-value" not in out


def test_a_body_that_is_not_an_object_is_said_so(capsys):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import BaseModel

    class Body(BaseModel):
        name: str

    probe = FastAPI()
    probe.add_exception_handler(
        __import__("fastapi.exceptions", fromlist=["x"]).RequestValidationError,
        app_mod.validation_error_says_which_field)

    @probe.post("/t")
    def _route(body: Body):
        return {"ok": True}

    client = TestClient(probe)
    r = client.post("/t", json=["not", "an", "object"])
    assert r.status_code == 422
    out = capsys.readouterr().out
    assert "422 on POST /t" in out


def test_the_real_app_actually_registers_it():
    """The lesson from the 500 handler next door: a handler keyed to the wrong
    thing is never called, and nothing says so. FastAPI raises
    RequestValidationError, NOT HTTPException(422), so a handler registered
    on the integer 422 would never fire."""
    from fastapi.exceptions import RequestValidationError
    handlers = getattr(app_mod.app, "exception_handlers", {})
    assert RequestValidationError in handlers, sorted(map(str, handlers))
    assert handlers[RequestValidationError] is \
        app_mod.validation_error_says_which_field
