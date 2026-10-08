"""Bounded single seat completion (``POST /v1/seat/completions``) behaviour contracts.

Offline only: the provider SDK clients are fixtures that record the exact kwargs handed to the SDK, so
these tests assert what would reach the wire (one call, no tools, provider-enforced output cap) and
that every refused admission is decided before any provider client exists.
"""

import hashlib
import json
import logging
import secrets
import sqlite3
import time
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server_seat_completion as seat
from gateway.platforms.api_server import APIServerAdapter

API_KEY = "seat-test-api-key-0123456789abcdef"
SECRET = secrets.token_hex(32)
PROMPT = "PRIVATE-MATTER-EVIDENCE: the lease at 12 Rue Example terminates on 1 March."
INSTRUCTION = "Return ONLY JSON. Use only the supplied evidence."


class FakeSDK:
    """Records each client construction and each create() call; returns a scripted response."""

    def __init__(self, response=None, error=None):
        self.clients, self.calls = [], []
        self.response, self.error = response, error

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _chat_response(text='{"ok": true}', finish="stop", out_tokens=12, tool_calls=None):
    message = SimpleNamespace(content=text, tool_calls=tool_calls, function_call=None)
    return SimpleNamespace(id="chatcmpl-fixture-1", choices=[SimpleNamespace(message=message, finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=40, completion_tokens=out_tokens))


def _anthropic_response(text='{"ok": true}', stop="end_turn"):
    return SimpleNamespace(id="msg_fixture_1", content=[SimpleNamespace(type="text", text=text)], stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=40, output_tokens=9))


def _responses_response(text='{"ok": true}', status="completed"):
    item = SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])
    return SimpleNamespace(id="resp_fixture_1", output=[item], output_text=text, status=status,
                           usage=SimpleNamespace(input_tokens=40, output_tokens=7))


RUNTIMES = {
    "chat_completions": {"provider": "custom", "api_mode": "chat_completions", "base_url": "https://llm.example/v1",
                         "api_key": "sk-fixture", "command": None, "args": [], "model": "owner-configured-model"},
    "anthropic_messages": {"provider": "anthropic", "api_mode": "anthropic_messages", "base_url": "",
                           "api_key": "sk-ant-api-fixture", "command": None, "args": [], "model": "claude-fixture"},
    "codex_responses": {"provider": "openai", "api_mode": "codex_responses", "base_url": "https://api.openai.com/v1",
                        "api_key": "sk-fixture", "command": None, "args": [], "model": "gpt-fixture"},
}


@pytest.fixture
def sdk(monkeypatch):
    """Patch the real SDK constructors the module imports at call time."""
    fake = FakeSDK(response=_chat_response())

    def openai_ctor(**kwargs):
        fake.clients.append(("openai", kwargs))
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake.create)),
                               responses=SimpleNamespace(create=fake.create))

    def anthropic_ctor(api_key, base_url=None, timeout=None, **_):
        fake.clients.append(("anthropic", {"api_key": api_key, "base_url": base_url, "timeout": timeout}))
        return SimpleNamespace(messages=SimpleNamespace(create=fake.create))

    import openai
    from agent import anthropic_adapter
    monkeypatch.setattr(openai, "OpenAI", openai_ctor)
    monkeypatch.setattr(anthropic_adapter, "build_anthropic_client", anthropic_ctor)
    return fake


@pytest.fixture
def runtime(monkeypatch):
    holder = {"value": dict(RUNTIMES["chat_completions"])}
    monkeypatch.setattr(seat, "resolve_seat_runtime", lambda: dict(holder["value"]) if holder["value"] else None)
    return holder


@pytest.fixture(autouse=True)
def _no_agent_loop(monkeypatch):
    """Any agent, tool dispatch or memory construction is a contract violation."""
    def boom(*_a, **_k):
        raise AssertionError("bounded seat completion must not construct an agent, tool call or memory")
    import model_tools
    import run_agent
    from agent import memory_manager
    monkeypatch.setattr(run_agent.AIAgent, "__init__", boom)
    monkeypatch.setattr(model_tools, "handle_function_call", boom)
    monkeypatch.setattr(memory_manager.MemoryManager, "__init__", boom)
    monkeypatch.setattr(APIServerAdapter, "_create_agent", boom)
    monkeypatch.setattr(APIServerAdapter, "_run_agent", boom)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("SEAT_ADMISSION_SECRET", SECRET)
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))


def _app(adapter):
    app = web.Application()
    for method, path, handler in seat._http_routes(adapter):
        app.router.add_route(method, path, handler)
    return app


def _binding(**overrides):
    now = int(time.time())
    binding = {"tenant_id": "tenant-a", "matter_id": "matter-1", "principal_id": "user-7",
               "task_id": "6f1b4f5c-1e0a-4a43-9a51-0d2c6b0f3e11", "run_id": "run-42",
               "capability_id": "research.conductor_run", "capability_version": 1, "audience": "private",
               "admission_key": hashlib.sha256(secrets.token_bytes(16)).hexdigest(),
               "admitted_cost_microunits": 25_000, "issued_at": now, "expires_at": now + 30}
    binding.update(overrides)
    return binding


def _body(binding=None, prompt=PROMPT, instruction=INSTRUCTION, max_out=4096, secret=SECRET, **mutate):
    binding = binding or _binding()
    seat_input = {"instruction": instruction, "prompt": prompt}
    digest = seat.input_digest(seat_input)
    body = {"contract": seat.CONTRACT, "binding": binding, "input": seat_input, "input_sha256": digest,
            "max_output_tokens": max_out,
            "admission_signature": seat.admission_signature(secret, binding=binding, input_sha256=digest,
                                                            max_output_tokens=max_out)}
    body.update(mutate)
    return body


async def _post(adapter, body, *, auth=True):
    headers = {"Authorization": f"Bearer {API_KEY}"} if auth else {}
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/v1/seat/completions", data=json.dumps(body), headers=headers)
        return resp.status, await resp.json()


async def _get(adapter, path):
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.get(path, headers={"Authorization": f"Bearer {API_KEY}"})
        return resp.status, await resp.json()


def _ledger_rows():
    path = seat.ledger_path()
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("SELECT * FROM seat_completion").fetchall()
    finally:
        conn.close()


FORBIDDEN_WIRE_KEYS = {"tools", "tool_choice", "functions", "function_call", "parallel_tool_calls",
                       "previous_response_id", "conversation", "background"}


# -- the one admitted invocation ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_route_makes_exactly_one_bounded_tool_free_call(adapter, sdk, runtime):
    status, payload = await _post(adapter, _body(max_out=2048))
    assert status == 200, payload
    assert payload["output_text"] == '{"ok": true}' and payload["state"] == "completed"
    assert len(sdk.calls) == 1 and len(sdk.clients) == 1
    kind, client_kwargs = sdk.clients[0]
    assert kind == "openai" and client_kwargs["max_retries"] == 0
    assert client_kwargs["base_url"] == "https://llm.example/v1"
    call = sdk.calls[0]
    assert call["model"] == "owner-configured-model"
    assert call["max_tokens"] == 2048 and call["n"] == 1 and call["stream"] is False
    assert not FORBIDDEN_WIRE_KEYS & set(call)
    assert call["messages"] == [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": PROMPT}]


@pytest.mark.asyncio
async def test_anthropic_route_caps_output_and_disables_thinking(adapter, sdk, runtime):
    runtime["value"] = dict(RUNTIMES["anthropic_messages"])
    sdk.response = _anthropic_response()
    status, payload = await _post(adapter, _body())
    assert status == 200, payload
    assert len(sdk.calls) == 1 and sdk.clients[0][0] == "anthropic"
    call = sdk.calls[0]
    assert call["model"] == "claude-fixture" and call["max_tokens"] == 4096
    assert call.get("thinking") in (None, {"type": "disabled"})
    assert not FORBIDDEN_WIRE_KEYS & set(call)
    assert [m["role"] for m in call["messages"]] == ["user"]


@pytest.mark.asyncio
async def test_responses_route_sends_max_output_tokens_and_stores_nothing(adapter, sdk, runtime):
    runtime["value"] = dict(RUNTIMES["codex_responses"])
    sdk.response = _responses_response()
    status, payload = await _post(adapter, _body(max_out=1000))
    assert status == 200, payload
    call = sdk.calls[0]
    assert call["max_output_tokens"] == 1000 and call["store"] is False and call["stream"] is False
    assert call["instructions"] == INSTRUCTION and not FORBIDDEN_WIRE_KEYS & set(call)


@pytest.mark.asyncio
async def test_request_cannot_choose_provider_model_tools_or_callback(adapter, sdk, runtime):
    for extra in ({"model": "other"}, {"provider": "openrouter"}, {"tools": [{"type": "function"}]},
                  {"callback_url": "https://evil.example"}, {"price_microunits": 0}, {"stream": True}):
        body = _body()
        body.update(extra)
        status, payload = await _post(adapter, body)
        assert (status, payload["error"]["code"]) == (400, "malformed_seat_request"), extra
    assert sdk.calls == [] and sdk.clients == [] and _ledger_rows() == []


# -- refusals before any model call -------------------------------------------------------------------

def _resign(body, binding):
    return _body(binding=binding, prompt=body["input"]["prompt"], instruction=body["input"]["instruction"],
                 max_out=body["max_output_tokens"])


def _tampered(field, value):
    body = _body()
    body["binding"][field] = value  # changed after signing
    return body


def _without_run_id():
    binding = _binding()
    del binding["run_id"]
    return _body(binding=binding)


def _tampered_prompt():
    body = _body()
    body["input"]["prompt"] = PROMPT + " tampered"
    return body


def _wrong_contract():
    body = _body()
    body["contract"] = "hermes.seat_completion.v0"
    return body


def _at(offset_issued, offset_expires):
    now = int(time.time())
    return _body(binding=_binding(issued_at=now + offset_issued, expires_at=now + offset_expires))


REFUSALS = {
    "unsigned_scope_change": (lambda: _tampered("matter_id", "matter-2"), (403, "admission_not_authorized")),
    "wrong_secret": (lambda: _body(secret=secrets.token_hex(32)), (403, "admission_not_authorized")),
    "expired": (lambda: _at(-90, -30), (403, "admission_expired")),
    "ttl_too_long": (lambda: _at(0, seat.MAX_ADMISSION_TTL_S + 1), (403, "admission_expired")),
    "future_issue": (lambda: _at(120, 150), (403, "admission_expired")),
    "shared_audience": (lambda: _body(binding=_binding(audience="matter")), (403, "audience_not_authorized")),
    "negative_cost": (lambda: _body(binding=_binding(admitted_cost_microunits=-1)), (400, "cost_admission_invalid")),
    "float_cost": (lambda: _body(binding=_binding(admitted_cost_microunits=1.5)), (400, "cost_admission_invalid")),
    "output_over_cap": (lambda: _body(max_out=seat.MAX_OUTPUT_TOKENS + 1), (400, "output_bound_invalid")),
    "input_over_cap": (lambda: _body(prompt="é" * (seat.MAX_INPUT_BYTES // 2 + 1)), (413, "input_bound_exceeded")),
    "digest_mismatch": (_tampered_prompt, (400, "input_digest_mismatch")),
    "missing_binding_field": (_without_run_id, (400, "malformed_seat_binding")),
    "bad_admission_key": (lambda: _body(binding=_binding(admission_key="not-a-digest")), (400, "malformed_seat_binding")),
    "wrong_contract": (_wrong_contract, (400, "malformed_seat_request")),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(REFUSALS))
async def test_invalid_admissions_refuse_before_model_call(adapter, sdk, runtime, case):
    make_body, expected = REFUSALS[case]
    status, payload = await _post(adapter, make_body())
    assert (status, payload["error"]["code"]) == expected
    assert sdk.calls == [] and sdk.clients == [] and _ledger_rows() == []


@pytest.mark.asyncio
async def test_input_at_exact_byte_bound_is_admitted(adapter, sdk, runtime):
    prompt = "x" * (seat.MAX_INPUT_BYTES - len(INSTRUCTION.encode()))
    status, _ = await _post(adapter, _body(prompt=prompt))
    assert status == 200 and len(sdk.calls) == 1


@pytest.mark.asyncio
async def test_gateway_bearer_is_required(adapter, sdk, runtime):
    status, _ = await _post(adapter, _body(), auth=False)
    assert status == 401 and sdk.calls == []


@pytest.mark.asyncio
async def test_missing_admission_secret_is_unavailable(adapter, sdk, runtime, monkeypatch):
    monkeypatch.delenv("SEAT_ADMISSION_SECRET")
    status, payload = await _post(adapter, _body())
    assert (status, payload["error"]["code"]) == (503, "seat_contract_unavailable") and sdk.calls == []


# -- unsupported routes fail closed, discovery included -----------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("change,reason", [
    ({"provider": "openai-codex", "api_mode": "codex_responses",
      "base_url": "https://chatgpt.com/backend-api/codex"}, "provider_output_bound_unenforceable"),
    ({"api_mode": "bedrock_converse"}, "api_mode_unsupported"),
    ({"api_mode": "codex_app_server", "command": "codex", "args": ["app-server"]},
     "external_process_runtime_unsupported"),
    ({"api_key": ""}, "seat_credential_unavailable"),
])
async def test_unsupported_seat_routes_never_call_or_fall_back(adapter, sdk, runtime, change, reason):
    runtime["value"] = {**RUNTIMES["chat_completions"], **change}
    status, payload = await _post(adapter, _body())
    assert (status, payload["error"]["code"]) == (503, reason)
    status, contract = await _get(adapter, "/v1/seat/contract")
    assert status == 200 and contract["available"] is False and contract["reason"] == reason
    assert sdk.calls == [] and sdk.clients == [] and _ledger_rows() == []


@pytest.mark.asyncio
async def test_discovery_reports_bounds_only_when_contract_is_real(adapter, sdk, runtime, monkeypatch):
    status, contract = await _get(adapter, "/v1/seat/contract")
    assert status == 200 and contract["available"] is True and contract["api_mode"] == "chat_completions"
    assert (contract["max_input_bytes"], contract["max_output_tokens"], contract["tools"],
            contract["invocations_per_admission"]) == (200_000, 4096, [], 1)
    monkeypatch.delenv("SEAT_ADMISSION_SECRET")
    _, contract = await _get(adapter, "/v1/seat/contract")
    assert contract["available"] is False and contract["reason"] == "seat_admission_secret_unconfigured"
    assert sdk.calls == []


def test_runtime_is_the_profile_configured_provider_without_fallback(monkeypatch):
    from hermes_cli import runtime_provider
    seen = {}

    def resolve(*, requested=None, target_model=None, **_):
        seen.update(requested=requested, target_model=target_model)
        return {"provider": "anthropic", "api_mode": "anthropic_messages", "base_url": "", "api_key": "k"}
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {"provider": "anthropic", "default": "m-1"})
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve)
    resolved = seat.resolve_seat_runtime()
    assert seen == {"requested": "anthropic", "target_model": "m-1"}
    assert resolved["model"] == "m-1" and resolved["api_mode"] == "anthropic_messages"


# -- durable outcomes and no resubmission -------------------------------------------------------------

@pytest.mark.asyncio
async def test_repeated_admission_key_never_invokes_again(adapter, sdk, runtime):
    body = _body()
    assert (await _post(adapter, body))[0] == 200
    status, payload = await _post(adapter, body)
    assert (status, payload["error"]["code"]) == (409, "admission_already_used")
    assert payload["outcome"]["state"] == "completed" and payload["outcome"]["resubmission_allowed"] is False
    other = _resign(body, {**body["binding"], "run_id": "run-43"})
    status, payload = await _post(adapter, other)
    assert (status, payload["error"]["code"]) == (409, "admission_key_conflict")
    assert len(sdk.calls) == 1


@pytest.mark.asyncio
async def test_lost_provider_response_is_unknown_and_not_resubmitted(adapter, sdk, runtime):
    sdk.error = ConnectionError("socket closed after send")
    body = _body()
    status, payload = await _post(adapter, body)
    assert (status, payload["error"]["code"]) == (502, "provider_outcome_unknown")
    assert payload["outcome"]["state"] == "unknown"
    sdk.error, sdk.response = None, _chat_response()
    status, payload = await _post(adapter, body)
    assert status == 409 and payload["outcome"]["state"] == "unknown"
    status, readback = await _get(adapter, f"/v1/seat/completions/{body['binding']['admission_key']}")
    assert status == 200 and readback["state"] == "unknown" and readback["resubmission_allowed"] is False
    assert len(sdk.calls) == 1


@pytest.mark.asyncio
async def test_definite_provider_rejection_is_failed_not_unknown(adapter, sdk, runtime):
    err = RuntimeError("bad request")
    err.status_code = 400
    sdk.error = err
    status, payload = await _post(adapter, _body())
    assert (status, payload["error"]["code"]) == (502, "provider_rejected")
    assert payload["outcome"]["state"] == "failed" and len(sdk.calls) == 1


def test_worker_settles_its_outcome_even_if_the_http_waiter_is_gone(sdk, runtime):
    """dispatch_and_settle owns the settlement, so cancellation of the request task cannot lose it."""
    binding = _binding()
    rt = dict(RUNTIMES["chat_completions"])
    assert seat.claim(binding["admission_key"], "b" * 64, rt, 1) is None
    state, _ = seat.dispatch_and_settle(binding["admission_key"], rt, {"instruction": "", "prompt": "p"}, 100)
    assert state == "completed" and seat.readback(binding["admission_key"])["state"] == "completed"
    assert seat.claim(binding["admission_key"], "b" * 64, rt, 1)["state"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    _chat_response(finish="length"),
    _chat_response(out_tokens=5000),
    _chat_response(finish="tool_calls", tool_calls=[{"id": "t"}]),
])
async def test_truncated_tool_or_overbound_output_is_withheld(adapter, sdk, runtime, response):
    sdk.response = response
    status, payload = await _post(adapter, _body())
    assert (status, payload["error"]["code"]) == (502, "provider_output_incomplete")
    assert "output_text" not in payload and payload["outcome"]["state"] == "incomplete"
    assert len(sdk.calls) == 1


@pytest.mark.asyncio
async def test_ledger_and_logs_hold_no_legal_prose_or_scope_ids(adapter, sdk, runtime, caplog):
    caplog.set_level(logging.DEBUG)
    body = _body()
    assert (await _post(adapter, body))[0] == 200
    dump = json.dumps(_ledger_rows()) + caplog.text
    for secret_text in (PROMPT, INSTRUCTION, '{"ok": true}', "tenant-a", "matter-1", "user-7", SECRET, "sk-fixture"):
        assert secret_text not in dump
    assert str(seat.ledger_path()).startswith(str(__import__("hermes_constants").get_hermes_home()))


# -- the runtime bound itself -------------------------------------------------------------------------

@pytest.mark.parametrize("mode,request_kwargs", [
    ("chat_completions", {"model": "m", "messages": [], "n": 1}),
    ("chat_completions", {"model": "m", "messages": [], "max_tokens": 9000}),
    ("chat_completions", {"model": "m", "messages": [], "max_tokens": 10, "tools": []}),
    ("chat_completions", {"model": "m", "messages": [], "max_tokens": 10, "stream": True}),
    ("anthropic_messages", {"model": "m", "messages": [], "max_tokens": 10, "thinking": {"type": "enabled"}}),
    ("codex_responses", {"model": "m", "input": [], "max_tokens": 10}),
])
def test_unbounded_request_never_reaches_sdk(mode, request_kwargs):
    with pytest.raises(RuntimeError, match="not bounded"):
        seat._assert_bounded(mode, request_kwargs, 4096)


def test_route_is_registered_beside_unchanged_chat_completions():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))
    table = {(m, p): h for m, p, h in adapter._http_route_table()}
    assert ("POST", "/v1/seat/completions") in table and ("GET", "/v1/seat/contract") in table
    assert table[("POST", "/v1/chat/completions")] == adapter._handle_chat_completions
