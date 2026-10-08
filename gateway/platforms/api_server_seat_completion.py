"""Bounded single seat completion: ``POST /v1/seat/completions`` (contract ``hermes.seat_completion.v1``).

A code-owned model-call contract beside, not inside, the agent-chat handlers. One admitted request makes
EXACTLY ONE provider invocation on the profile's own configured provider/model: no AIAgent, no tool
loop, no tools on the wire, no continuation/retry, no session/transcript, no memory/skills/context
files. The caller supplies only an instruction + prompt (the authorized task evidence); it cannot pick
a provider, model, price, tool, callback or stream.

Authority is canonical and lives elsewhere. The product (legal-api) admits the call against its existing
delegated grant + inherited budget (one row, ``terminal_delegated_grants``) and signs the binding with the
profile-scoped secret ``SEAT_ADMISSION_SECRET``. A signature is NOT authority on its own: immediately before
the SDK is constructed, inside the dispatch worker, this module asks the operator-configured canonical
authority (``SEAT_AUTHORITY_URL``, never a caller-supplied URL) to verify and consume that exact admission
against the CURRENT grant row (revocation, expiry, issuer session, scope, task/run pin, admitted cost,
input digest, output bound, cancellation, one-time dispatch). Only a signed ``admitted`` verdict bound to a
fresh nonce lets the call proceed, and the admission expiry is re-checked after that verdict. This module
stores no grant or budget: it records a metadata-only outcome keyed by the admission key so a lost response
is never re-invoked (a repeat key answers 409 with the recorded state).

Privacy: during the single invocation (and the authority exchange) every record from the provider SDK and
HTTP-stack loggers (openai*, anthropic*, httpx*, httpcore*) is dropped by a filter keyed on a context
variable that is set only in the dispatching worker, so request/response/error bodies never reach logs
while concurrent ordinary traffic in other threads/tasks keeps its normal logging.

Bounds are enforced where they cost money: the provider request carries the output cap
(``max_tokens`` / ``max_output_tokens``) and no tools, asserted on the final kwargs immediately before
the SDK call. A route whose provider cannot enforce the cap (ChatGPT Codex backend) or whose api_mode
this module does not drive is reported unavailable; nothing falls back to the agent loop.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import hmac
import json
import logging
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    from aiohttp import web
except ImportError:  # pragma: no cover - aiohttp is a gateway dependency
    web = None  # type: ignore[assignment]

logger = logging.getLogger("gateway.platforms.api_server.seat_completion")

CONTRACT = "hermes.seat_completion.v1"
MAX_INPUT_BYTES = 200_000
MAX_OUTPUT_TOKENS = 4096
MAX_ADMISSION_TTL_S = 60
CLOCK_SKEW_S = 5
# JSON escaping can inflate the 200 KB UTF-8 input; the decoded input is bounded exactly below.
MAX_BODY_BYTES = 1_400_000
PROVIDER_TIMEOUT_S = 300.0
SECRET_NAME = "SEAT_ADMISSION_SECRET"
AUTHORITY_URL_NAME = "SEAT_AUTHORITY_URL"
MIN_SECRET_LEN = 32
AUTHORITY_TIMEOUT_S = 10.0
# Domain separation: an admission signature can never be replayed as an authority request or verdict.
ADMISSION_DOMAIN = CONTRACT + "/admission"
AUTHORITY_REQUEST_DOMAIN = CONTRACT + "/authority-request"
AUTHORITY_VERDICT_DOMAIN = CONTRACT + "/authority-verdict"
_PRIVATE_LOGGER_PREFIXES = ("openai", "anthropic", "httpx", "httpcore")

_HEX64 = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9._:@/-]{1,200}")
_BODY_KEYS = frozenset({"contract", "binding", "input", "input_sha256", "max_output_tokens", "admission_signature"})
_BINDING_KEYS = frozenset({
    "tenant_id", "matter_id", "principal_id", "task_id", "run_id", "capability_id", "capability_version",
    "delegation_ref", "operation_id", "seat_route_sha256", "audience", "admission_key", "admitted_cost_microunits", "issued_at",
    "expires_at"})
_INPUT_KEYS = frozenset({"instruction", "prompt"})
_CODEX_BACKEND_MARKER = "chatgpt.com/backend-api/codex"


class SeatRefused(Exception):
    """A refusal decided before any provider invocation (no ledger row, no model call)."""

    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status, self.code = status, code


def canonical(value: Any) -> bytes:
    """The signed byte form; legal-api must produce the identical bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def input_digest(seat_input: Dict[str, str]) -> str:
    return hashlib.sha256(canonical(seat_input)).hexdigest()


def sign(secret: str, domain: str, payload: Dict[str, Any]) -> str:
    """HMAC-SHA256 over ``canonical({"domain": domain, "payload": payload})``."""
    return hmac.new(secret.encode("utf-8"), canonical({"domain": domain, "payload": payload}),
                    hashlib.sha256).hexdigest()


def admission_signature(secret: str, *, binding: Dict[str, Any], input_sha256: str,
                        max_output_tokens: int) -> str:
    payload = {"contract": CONTRACT, "binding": binding, "input_sha256": input_sha256,
               "max_output_tokens": max_output_tokens}
    return sign(secret, ADMISSION_DOMAIN, payload)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _admission_secret() -> str:
    """Profile-scoped secret; empty (unavailable) when missing or weak. Never logged."""
    from agent.secret_scope import UnscopedSecretError, get_secret
    from hermes_cli.auth import has_usable_secret
    try:
        value = get_secret(SECRET_NAME, "") or ""
    except UnscopedSecretError:
        logger.warning("seat completion secret unavailable: no profile scope bound")
        return ""
    return value if has_usable_secret(value, min_length=MIN_SECRET_LEN) else ""


def _authority_url() -> str:
    """Operator-configured canonical authority endpoint (profile-scoped); empty when absent/invalid.

    It is configuration, never request data: the caller cannot name a callback.
    """
    from agent.secret_scope import UnscopedSecretError, get_secret
    try:
        value = (get_secret(AUTHORITY_URL_NAME, "") or "").strip()
    except UnscopedSecretError:
        return ""
    if not re.fullmatch(r"https?://[A-Za-z0-9.\-\[\]:]+(/[A-Za-z0-9._~/\-]*)?", value):
        return ""
    return value


# -- request validation (pure; every refusal happens before the ledger or the provider) -----------------

def _validate_shape(body: Any) -> Tuple[Dict[str, Any], Dict[str, str], int]:
    if type(body) is not dict or set(body) != _BODY_KEYS or body["contract"] != CONTRACT:
        raise SeatRefused(400, "malformed_seat_request")
    binding, seat_input, max_out = body["binding"], body["input"], body["max_output_tokens"]
    if type(binding) is not dict or set(binding) != _BINDING_KEYS:
        raise SeatRefused(400, "malformed_seat_binding")
    if type(seat_input) is not dict or set(seat_input) != _INPUT_KEYS or not all(
            isinstance(seat_input[k], str) for k in _INPUT_KEYS) or not seat_input["prompt"].strip():
        raise SeatRefused(400, "malformed_seat_input")
    if not _is_int(max_out) or not 1 <= max_out <= MAX_OUTPUT_TOKENS:
        raise SeatRefused(400, "output_bound_invalid")
    for key in ("tenant_id", "matter_id", "principal_id", "task_id", "run_id", "capability_id", "operation_id"):
        if not isinstance(binding[key], str) or not _ID.fullmatch(binding[key]):
            raise SeatRefused(400, "malformed_seat_binding")
    if not _is_int(binding["capability_version"]) or binding["capability_version"] < 1:
        raise SeatRefused(400, "malformed_seat_binding")
    if not isinstance(binding["admission_key"], str) or not _HEX64.fullmatch(binding["admission_key"]):
        raise SeatRefused(400, "malformed_seat_binding")
    if not isinstance(binding["seat_route_sha256"], str) or not _HEX64.fullmatch(binding["seat_route_sha256"]):
        raise SeatRefused(400, "malformed_seat_binding")
    ref = binding["delegation_ref"]  # opaque one-way grant reference; the authority resolves it, Hermes never does
    if not isinstance(ref, str) or not ref.startswith("grant-sha256:") or not _HEX64.fullmatch(ref[13:]):
        raise SeatRefused(400, "malformed_seat_binding")
    cost = binding["admitted_cost_microunits"]
    if not _is_int(cost) or not 1 <= cost < 10 ** 15:
        raise SeatRefused(400, "cost_admission_invalid")
    for key in ("issued_at", "expires_at"):
        if not _is_int(binding[key]) or binding[key] <= 0:
            raise SeatRefused(400, "malformed_seat_binding")
    if not isinstance(body["input_sha256"], str) or not _HEX64.fullmatch(body["input_sha256"]):
        raise SeatRefused(400, "malformed_seat_request")
    if not isinstance(body["admission_signature"], str) or not _HEX64.fullmatch(body["admission_signature"]):
        raise SeatRefused(400, "malformed_seat_request")
    return binding, seat_input, max_out


def validate_request(body: Any, *, secret: str, now: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, str], int]:
    """Exact schema, signature, scope, TTL and input bounds. Returns (binding, input, max_output_tokens)."""
    binding, seat_input, max_out = _validate_shape(body)
    if not secret:
        raise SeatRefused(503, "seat_contract_unavailable")
    expected = admission_signature(secret, binding=binding, input_sha256=body["input_sha256"],
                                   max_output_tokens=max_out)
    if not hmac.compare_digest(expected, body["admission_signature"]):
        raise SeatRefused(403, "admission_not_authorized")
    if binding["audience"] != "private":
        raise SeatRefused(403, "audience_not_authorized")
    check_admission_window(binding, now)
    size = sum(len(seat_input[k].encode("utf-8")) for k in _INPUT_KEYS)
    if size > MAX_INPUT_BYTES:
        raise SeatRefused(413, "input_bound_exceeded")
    if not hmac.compare_digest(input_digest(seat_input), body["input_sha256"]):
        raise SeatRefused(400, "input_digest_mismatch")
    return binding, seat_input, max_out


def check_admission_window(binding: Dict[str, Any], now: Optional[float] = None) -> None:
    """TTL window. Called at request validation AND again at dispatch, after every wait."""
    current = time.time() if now is None else now
    issued, expires = binding["issued_at"], binding["expires_at"]
    if (issued > current + CLOCK_SKEW_S or expires <= current or expires <= issued
            or expires - issued > MAX_ADMISSION_TTL_S):
        raise SeatRefused(403, "admission_expired")


# -- configured seat runtime --------------------------------------------------------------------------

def _is_codex_backend(runtime: Dict[str, Any]) -> bool:
    return (runtime.get("provider") == "openai-codex"
            or _CODEX_BACKEND_MARKER in str(runtime.get("base_url") or "").lower())


def unsupported_reason(runtime: Optional[Dict[str, Any]]) -> Optional[str]:
    """None when this module can drive the route with a provider-enforced output cap."""
    if not runtime:
        return "seat_runtime_unresolved"
    if runtime.get("command") or runtime.get("args"):
        return "external_process_runtime_unsupported"
    if not isinstance(runtime.get("api_key"), str) or not runtime["api_key"]:
        return "seat_credential_unavailable"
    if not isinstance(runtime.get("model"), str) or not runtime["model"]:
        return "seat_model_unconfigured"
    mode = runtime.get("api_mode")
    if mode == "codex_responses" and _is_codex_backend(runtime):
        # The ChatGPT Codex backend rejects max_output_tokens: the cap would be advisory only.
        return "provider_output_bound_unenforceable"
    if mode not in _DRIVERS:
        return "api_mode_unsupported"
    return None


def seat_route(runtime: Dict[str, Any]) -> Dict[str, str]:
    """The owner-configured route an admission is bound to (no credential, no URL secrets)."""
    return {"provider": str(runtime.get("provider") or ""), "api_mode": str(runtime.get("api_mode") or ""),
            "model": str(runtime.get("model") or "")}


def seat_route_sha256(runtime: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical(seat_route(runtime))).hexdigest()


def resolve_seat_runtime() -> Optional[Dict[str, Any]]:
    """The profile's own configured provider/model; no fallback chain, no request override."""
    from hermes_cli.runtime_provider import _get_model_config, resolve_runtime_provider
    cfg = _get_model_config() or {}
    provider = str(cfg.get("provider") or "").strip()
    model = str(cfg.get("default") or "").strip()
    if not provider or not model:
        return None
    runtime = resolve_runtime_provider(requested=provider, target_model=model)
    return {"provider": runtime.get("provider"), "api_mode": runtime.get("api_mode"),
            "base_url": runtime.get("base_url") or "", "api_key": runtime.get("api_key"),
            "command": runtime.get("command"), "args": list(runtime.get("args") or []), "model": model}


def _resolve_or_none() -> Optional[Dict[str, Any]]:
    """Unresolvable seat runtime is reported unavailable, never an agent-chat fallback."""
    try:
        return resolve_seat_runtime()
    # health: allow BLE001 -- availability boundary over the whole provider-resolution ladder (auth, config, pool errors); every failure means "unavailable" and only the type is logged because resolver messages can name credential sources
    except Exception as exc:
        logger.warning("seat runtime resolution failed: %s", type(exc).__name__)
        return None


# -- privacy at the SDK / HTTP-stack boundary ----------------------------------------------------------

# True only inside the worker that performs the authority exchange and the one provider invocation.
# asyncio.to_thread copies the context, so setting it there never leaks into other tasks or threads.
_PRIVATE_CALL: contextvars.ContextVar[bool] = contextvars.ContextVar("seat_private_call", default=False)


class _PrivateCallFilter(logging.Filter):
    """Drops SDK/HTTP-stack records emitted by the private call's own thread/context only."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (_PRIVATE_CALL.get() and record.name.split(".", 1)[0] in _PRIVATE_LOGGER_PREFIXES)


_PRIVATE_FILTER = _PrivateCallFilter()
_guard_lock = threading.Lock()


def _guard_private_logging() -> None:
    """Attach the context-keyed filter to every SDK/HTTP logger and every root handler (idempotent).

    Logger filters cover records at their origin; handler filters cover propagated records from child
    loggers created after this scan. No level is changed, so ordinary traffic logs exactly as before.
    """
    with _guard_lock:
        manager = logging.Logger.manager
        names = [n for n in list(manager.loggerDict) if n.split(".", 1)[0] in _PRIVATE_LOGGER_PREFIXES]
        loggers = [logging.getLogger(n) for n in (*_PRIVATE_LOGGER_PREFIXES, *names)]
        handlers = list(logging.getLogger().handlers)
        if logging.lastResort is not None:  # used when no handler is configured at all
            handlers.append(logging.lastResort)
        for lg in loggers:
            handlers.extend(lg.handlers)
        for target in (*loggers, *handlers):
            if _PRIVATE_FILTER not in target.filters:
                target.addFilter(_PRIVATE_FILTER)


# -- canonical authority at dispatch -------------------------------------------------------------------

def _authority_client() -> Any:
    import httpx
    return httpx.Client(timeout=AUTHORITY_TIMEOUT_S, trust_env=False, follow_redirects=False)


def _refusal_code(value: Any) -> str:
    code = value.get("code") if isinstance(value, dict) else None
    return code if isinstance(code, str) and re.fullmatch(r"[a-z_]{1,64}", code) else "admission_not_authorized"


def authorize_at_dispatch(secret: str, url: str, *, binding: Dict[str, Any], input_sha256: str, max_out: int,
                          admission_signature_hex: str, runtime: Dict[str, Any]) -> None:
    """Ask the canonical product authority to verify AND consume this exact admission, now.

    The authority re-reads the current delegated grant (revocation, expiry, issuer session, scope, task/run
    pin, admitted cost, bound input digest/output cap, cancellation) and marks the admission dispatched
    once. Only a verdict signed over this request's digest and fresh nonce admits the call; anything else
    (refusal, transport loss, unreadable or unsigned reply) refuses BEFORE any SDK client exists.
    """
    import httpx
    request = {"contract": CONTRACT, "binding": binding, "input_sha256": input_sha256,
               "max_output_tokens": max_out, "admission_signature": admission_signature_hex,
               "nonce": secrets.token_hex(32), "requested_at": int(time.time()),
               "seat_route": seat_route(runtime)}
    body = canonical({"request": request, "request_signature": sign(secret, AUTHORITY_REQUEST_DOMAIN, request)})
    try:
        with _authority_client() as client:
            response = client.post(url, content=body, headers={"Content-Type": "application/json"})
        verdict = response.json()
    except (httpx.HTTPError, ValueError):
        raise SeatRefused(503, "seat_authority_unavailable") from None
    if response.status_code != 200:
        raise SeatRefused(503 if response.status_code >= 500 else 403, _refusal_code(verdict))
    expected = {"verdict": "admitted", "request_sha256": hashlib.sha256(canonical(request)).hexdigest(),
                "nonce": request["nonce"]}
    signature = verdict.get("authority_signature") if isinstance(verdict, dict) else None
    if (not isinstance(verdict, dict) or set(verdict) != {*expected, "authority_signature"}
            or any(verdict[k] != v for k, v in expected.items()) or not isinstance(signature, str)
            or not hmac.compare_digest(sign(secret, AUTHORITY_VERDICT_DOMAIN, expected), signature)):
        raise SeatRefused(403, "authority_verdict_invalid")


# -- the single invocation ----------------------------------------------------------------------------

def _anthropic_request(runtime: Dict[str, Any], seat_input: Dict[str, str], max_out: int) -> Dict[str, Any]:
    from agent.anthropic_adapter import build_anthropic_kwargs
    from agent.anthropic_credentials import anthropic_route_is_oauth
    messages = [{"role": "user", "content": seat_input["prompt"]}]
    if seat_input["instruction"]:
        messages.insert(0, {"role": "system", "content": seat_input["instruction"]})
    oauth = anthropic_route_is_oauth(runtime.get("base_url"), runtime["api_key"], provider=runtime.get("provider"))
    # Thinking disabled so max_tokens bounds every generated token.
    return build_anthropic_kwargs(model=runtime["model"], messages=messages, tools=None, max_tokens=max_out,
                                  reasoning_config={"enabled": False}, is_oauth=oauth,
                                  base_url=runtime.get("base_url") or None)


def _chat_request(runtime: Dict[str, Any], seat_input: Dict[str, str], max_out: int) -> Dict[str, Any]:
    messages = [{"role": "user", "content": seat_input["prompt"]}]
    if seat_input["instruction"]:
        messages.insert(0, {"role": "system", "content": seat_input["instruction"]})
    return {"model": runtime["model"], "messages": messages, "max_tokens": max_out, "n": 1, "stream": False}


def _responses_request(runtime: Dict[str, Any], seat_input: Dict[str, str], max_out: int) -> Dict[str, Any]:
    request = {"model": runtime["model"], "input": [{"role": "user", "content": seat_input["prompt"]}],
               "max_output_tokens": max_out, "store": False, "stream": False}
    if seat_input["instruction"]:
        request["instructions"] = seat_input["instruction"]
    return request


def _assert_bounded(api_mode: str, request: Dict[str, Any], max_out: int) -> None:
    """Last check on the exact SDK kwargs: one bounded, tool-free, non-streamed generation."""
    cap_key = {"anthropic_messages": "max_tokens", "chat_completions": "max_tokens",
               "codex_responses": "max_output_tokens"}[api_mode]
    forbidden = {"tools", "tool_choice", "functions", "function_call", "parallel_tool_calls",
                 "previous_response_id", "conversation", "background"}
    thinking = request.get("thinking")
    if (not _is_int(request.get(cap_key)) or not 1 <= request[cap_key] <= max_out
            or forbidden & set(request) or request.get("stream") not in (None, False)
            or request.get("n", 1) != 1
            or (thinking is not None and thinking != {"type": "disabled"})):
        raise RuntimeError("seat request is not bounded")


def _anthropic_client(runtime: Dict[str, Any]) -> Any:
    from agent.anthropic_adapter import build_anthropic_client
    return build_anthropic_client(runtime["api_key"], runtime.get("base_url") or None, timeout=PROVIDER_TIMEOUT_S)


def _openai_client(runtime: Dict[str, Any]) -> Any:
    from openai import OpenAI
    return OpenAI(api_key=runtime["api_key"], base_url=runtime.get("base_url") or None,
                  max_retries=0, timeout=PROVIDER_TIMEOUT_S)


def _usage(usage: Any, prompt_key: str, out_key: str) -> Dict[str, Optional[int]]:
    def read(key: str) -> Optional[int]:
        value = getattr(usage, key, None) if usage is not None else None
        return value if _is_int(value) and value >= 0 else None
    return {"input_tokens": read(prompt_key), "output_tokens": read(out_key)}


def _anthropic_result(response: Any) -> Dict[str, Any]:
    blocks = list(getattr(response, "content", None) or [])
    text = "".join(getattr(b, "text", "") or "" for b in blocks if getattr(b, "type", "") == "text")
    stop = getattr(response, "stop_reason", None)
    tool = any(getattr(b, "type", "") != "text" for b in blocks)
    return {"text": text, "complete": stop == "end_turn" and not tool, "finish": str(stop or "unknown"),
            "usage": _usage(getattr(response, "usage", None), "input_tokens", "output_tokens"),
            "provider_request_id": getattr(response, "id", None)}


def _chat_result(response: Any) -> Dict[str, Any]:
    choices = list(getattr(response, "choices", None) or [])
    choice = choices[0] if len(choices) == 1 else None
    message = getattr(choice, "message", None)
    finish = getattr(choice, "finish_reason", None)
    text = getattr(message, "content", None)
    tool = bool(getattr(message, "tool_calls", None) or getattr(message, "function_call", None))
    return {"text": text if isinstance(text, str) else "", "complete": finish == "stop" and not tool and choice is not None,
            "finish": str(finish or "unknown"),
            "usage": _usage(getattr(response, "usage", None), "prompt_tokens", "completion_tokens"),
            "provider_request_id": getattr(response, "id", None)}


def _responses_result(response: Any) -> Dict[str, Any]:
    items = list(getattr(response, "output", None) or [])
    tool = any(getattr(i, "type", "") not in ("message", "reasoning") for i in items)
    status = getattr(response, "status", None)
    text = getattr(response, "output_text", None)
    return {"text": text if isinstance(text, str) else "", "complete": status == "completed" and not tool,
            "finish": str(status or "unknown"),
            "usage": _usage(getattr(response, "usage", None), "input_tokens", "output_tokens"),
            "provider_request_id": getattr(response, "id", None)}


# api_mode -> (request builder, client factory, client -> create callable, result normalizer)
# Client factories are looked up at call time so one module-level seam builds every SDK client.
_DRIVERS: Dict[str, Tuple[Callable, Callable, Callable, Callable]] = {
    "anthropic_messages": (_anthropic_request, lambda rt: _anthropic_client(rt), lambda c: c.messages.create,
                           _anthropic_result),
    "chat_completions": (_chat_request, lambda rt: _openai_client(rt), lambda c: c.chat.completions.create,
                         _chat_result),
    "codex_responses": (_responses_request, lambda rt: _openai_client(rt), lambda c: c.responses.create,
                        _responses_result),
}


class ProviderOutcomeUnknown(Exception):
    """The request may have reached the provider; never resubmit."""


class ProviderRejected(Exception):
    """The provider answered with a definite HTTP error status before producing output."""


def invoke_once(runtime: Dict[str, Any], seat_input: Dict[str, str], max_out: int) -> Dict[str, Any]:
    """Exactly one SDK call. No retries here or in the client (max_retries=0)."""
    build, make_client, create, normalize = _DRIVERS[runtime["api_mode"]]
    request = build(runtime, seat_input, max_out)
    _assert_bounded(runtime["api_mode"], request, max_out)
    client = make_client(runtime)
    _guard_private_logging()  # after construction: every SDK/HTTP logger now exists
    try:
        response = create(client)(**request)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if _is_int(status) and 400 <= status < 500 and status not in (408, 409, 429):
            raise ProviderRejected(str(status)) from exc
        raise ProviderOutcomeUnknown(type(exc).__name__) from exc
    result = normalize(response)
    out_tokens = result["usage"]["output_tokens"]
    if out_tokens is not None and out_tokens > max_out:
        result["complete"] = False
        result["finish"] = "output_bound_exceeded"
    return result


# -- durable metadata-only outcome ledger -------------------------------------------------------------

_LEDGER_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS seat_completion ("
    "admission_key TEXT PRIMARY KEY, binding_sha256 TEXT NOT NULL, state TEXT NOT NULL, "
    "api_mode TEXT, model TEXT, admitted_cost_microunits INTEGER NOT NULL, finish TEXT, "
    "input_tokens INTEGER, output_tokens INTEGER, output_sha256 TEXT, provider_request_id TEXT, "
    "created_at REAL NOT NULL, updated_at REAL NOT NULL)")
_ledger_lock = threading.Lock()


def ledger_path() -> Path:
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home()) / "seat_completions.db"


def _connect() -> sqlite3.Connection:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(_LEDGER_SCHEMA)
    return conn


def _row(conn: sqlite3.Connection, key: str) -> Optional[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM seat_completion WHERE admission_key=?", (key,)).fetchone()
    return dict(row) if row else None


def claim(key: str, binding_sha256: str, runtime: Dict[str, Any], cost: int) -> Optional[Dict[str, Any]]:
    """Durably record ``dispatching`` BEFORE the provider call. Returns the existing row on a repeat."""
    now = time.time()
    with _ledger_lock:
        conn = _connect()
        try:
            try:
                conn.execute(
                    "INSERT INTO seat_completion (admission_key, binding_sha256, state, api_mode, model, "
                    "admitted_cost_microunits, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (key, binding_sha256, "dispatching", runtime["api_mode"], runtime["model"], cost, now, now))
                return None
            except sqlite3.IntegrityError:
                return _row(conn, key)
        finally:
            conn.close()


def settle(key: str, state: str, result: Optional[Dict[str, Any]] = None, *, finish: Optional[str] = None) -> None:
    usage = (result or {}).get("usage") or {}
    text = (result or {}).get("text")
    with _ledger_lock:
        conn = _connect()
        try:
            conn.execute(
                "UPDATE seat_completion SET state=?, finish=?, input_tokens=?, output_tokens=?, output_sha256=?, "
                "provider_request_id=?, updated_at=? WHERE admission_key=? AND state='dispatching'",
                (state, finish or (result or {}).get("finish"), usage.get("input_tokens"), usage.get("output_tokens"),
                 hashlib.sha256(text.encode("utf-8")).hexdigest() if isinstance(text, str) and result else None,
                 _bounded_id((result or {}).get("provider_request_id")), time.time(), key))
        finally:
            conn.close()


def _bounded_id(value: Any) -> Optional[str]:
    return value[:200] if isinstance(value, str) and value and not any(c.isspace() for c in value) else None


class Admission:
    """Everything the dispatch worker needs to re-prove authority after the request-time checks."""

    def __init__(self, *, binding: Dict[str, Any], input_sha256: str, max_out: int, signature: str,
                 secret: str, authority_url: str):
        self.binding, self.input_sha256, self.max_out = binding, input_sha256, max_out
        self.signature, self.secret, self.authority_url = signature, secret, authority_url


def dispatch_and_settle(key: str, runtime: Dict[str, Any], seat_input: Dict[str, str], max_out: int,
                        admission: Admission) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """For an already-claimed key: re-check the window, obtain the canonical authority's verdict, re-check
    the window again, then run the one invocation, and durably settle. Never retries.

    Runs in its own worker context: the private-logging flag set here is invisible to every other request.
    """
    token = _PRIVATE_CALL.set(True)
    try:
        return _dispatch_private(key, runtime, seat_input, max_out, admission)
    finally:
        _PRIVATE_CALL.reset(token)


def _dispatch_private(key: str, runtime: Dict[str, Any], seat_input: Dict[str, str], max_out: int,
                      admission: Admission) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    _guard_private_logging()
    result: Optional[Dict[str, Any]] = None
    try:
        # Admission time may have passed while runtime resolution, the claim or the thread queue waited.
        check_admission_window(admission.binding)
        authorize_at_dispatch(admission.secret, admission.authority_url, binding=admission.binding,
                              input_sha256=admission.input_sha256, max_out=max_out,
                              admission_signature_hex=admission.signature, runtime=runtime)
        check_admission_window(admission.binding)  # the authority exchange itself took time
    except SeatRefused as refused:
        # Definite: no provider client was constructed. The key stays burned (never re-dispatched).
        settle(key, "refused", finish=refused.code)
        _log(key, "refused", runtime)
        return "refused", None, refused.code
    try:
        result = invoke_once(runtime, seat_input, max_out)
        state = "completed" if result["complete"] else "incomplete"
    except ProviderRejected:
        state = "failed"
    # health: allow BLE001 -- settlement boundary: any failure after the claim may follow a sent request, so it is recorded as unknown (never retried); the traceback is not logged because provider error text can echo private matter content
    except Exception:
        state = "unknown"
    settle(key, state, result)
    _log(key, state, runtime, result)
    return state, result, None


def readback(key: str) -> Optional[Dict[str, Any]]:
    with _ledger_lock:
        conn = _connect()
        try:
            row = _row(conn, key)
        finally:
            conn.close()
    if row is None:
        return None
    state = row["state"]
    # A dispatch that outlived its provider timeout lost its process: report it as unknown, never as absent.
    if state == "dispatching" and time.time() - row["updated_at"] > PROVIDER_TIMEOUT_S + 60:
        state = "unknown"
    return {"contract": CONTRACT, "admission_key": row["admission_key"], "state": state,
            "binding_sha256": row["binding_sha256"], "api_mode": row["api_mode"], "model": row["model"],
            "admitted_cost_microunits": row["admitted_cost_microunits"], "finish": row["finish"],
            "usage": {"input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"]},
            "output_sha256": row["output_sha256"], "provider_request_id": row["provider_request_id"],
            # The output text is never retained; a completed call whose response was lost is unrecoverable.
            # "refused" is decided before any SDK client exists; every other state may have reached it.
            "provider_invoked": False if state == "refused" else None if state in ("dispatching", "unknown") else True,
            "output_retained": False, "resubmission_allowed": False}


# -- HTTP handlers ------------------------------------------------------------------------------------

def _error(status: int, code: str, **extra: Any) -> "web.Response":
    return web.json_response({"error": {"code": code, "type": "seat_completion_error"}, "contract": CONTRACT,
                              **extra}, status=status)


def _log(key: str, state: str, runtime: Optional[Dict[str, Any]] = None, result: Optional[Dict[str, Any]] = None) -> None:
    # Metadata only: never instruction, prompt, output, ids of tenant/matter/principal or credentials.
    usage = (result or {}).get("usage") or {}
    logger.info("seat completion key=%s state=%s api_mode=%s in=%s out=%s", key[:12], state,
                (runtime or {}).get("api_mode"), usage.get("input_tokens"), usage.get("output_tokens"))


def _reject_constant(_name: str) -> Any:
    raise ValueError("non-finite JSON number")


async def _read_bounded(request: "web.Request") -> Optional[bytes]:
    """Whole body, or None past MAX_BODY_BYTES (``content.read(n)`` returns one chunk, not n bytes)."""
    chunks, total = [], 0
    while True:
        chunk = await request.content.read(65536)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            return None
        chunks.append(chunk)


async def handle_seat_completion(adapter: Any, request: "web.Request") -> "web.Response":
    auth_err = adapter._check_auth(request)
    if auth_err is not None:
        return auth_err
    if request.content_length is not None and request.content_length > MAX_BODY_BYTES:
        return _error(413, "input_bound_exceeded")
    raw = await _read_bounded(request)
    if raw is None:
        return _error(413, "input_bound_exceeded")
    try:
        body = json.loads(raw, parse_constant=_reject_constant)
    except (ValueError, UnicodeDecodeError):
        return _error(400, "malformed_seat_request")
    secret = _admission_secret()
    try:
        binding, seat_input, max_out = validate_request(body, secret=secret)
        authority_url = _authority_url()
        if not authority_url:
            raise SeatRefused(503, "seat_contract_unavailable")
        runtime = await asyncio.to_thread(_resolve_or_none)
        reason = unsupported_reason(runtime)
        if reason is not None:
            raise SeatRefused(503, reason)
        if not hmac.compare_digest(seat_route_sha256(runtime), binding["seat_route_sha256"]):
            # The configured seat changed since the product admitted the call: never run it on another route.
            raise SeatRefused(409, "seat_route_changed")
    except SeatRefused as refused:
        return _error(refused.status, refused.code)
    key = binding["admission_key"]
    binding_sha256 = hashlib.sha256(canonical(binding)).hexdigest()
    existing = await asyncio.to_thread(claim, key, binding_sha256, runtime, binding["admitted_cost_microunits"])
    if existing is not None:
        conflict = "admission_key_conflict" if existing["binding_sha256"] != binding_sha256 else "admission_already_used"
        return _error(409, conflict, outcome=await asyncio.to_thread(readback, key))
    admission = Admission(binding=binding, input_sha256=body["input_sha256"], max_out=max_out,
                          signature=body["admission_signature"], secret=secret, authority_url=authority_url)
    # The worker thread re-proves authority and settles its own outcome, so neither a wait before dispatch
    # nor a client disconnect (task cancellation) can turn a stale admission into a call or lose a result.
    state, result, refusal = await asyncio.to_thread(dispatch_and_settle, key, runtime, seat_input, max_out,
                                                     admission)
    outcome = await asyncio.to_thread(readback, key)
    if state == "refused":
        return _error(503 if refusal == "seat_authority_unavailable" else 403, refusal or "admission_not_authorized",
                      outcome=outcome)
    if state == "failed":
        return _error(502, "provider_rejected", outcome=outcome)
    if state == "unknown":
        return _error(502, "provider_outcome_unknown", outcome=outcome)
    if state != "completed":
        # A truncated, tool-calling or over-bound generation is not an answer; its text is withheld.
        return _error(502, "provider_output_incomplete", outcome=outcome)
    return web.json_response({**(outcome or {}), "output_text": (result or {}).get("text", "")},
                             headers={"X-Hermes-Seat-Outcome": state})


async def handle_seat_readback(adapter: Any, request: "web.Request") -> "web.Response":
    auth_err = adapter._check_auth(request)
    if auth_err is not None:
        return auth_err
    key = request.match_info.get("admission_key", "")
    if not _HEX64.fullmatch(key):
        return _error(400, "malformed_admission_key")
    outcome = await asyncio.to_thread(readback, key)
    return web.json_response(outcome) if outcome else _error(404, "admission_not_recorded")


async def handle_seat_contract(adapter: Any, request: "web.Request") -> "web.Response":
    """Discovery: available only when the secret, the canonical authority and a cap-enforcing route all exist."""
    auth_err = adapter._check_auth(request)
    if auth_err is not None:
        return auth_err
    reason: Optional[str] = None if _admission_secret() else "seat_admission_secret_unconfigured"
    if reason is None and not _authority_url():
        reason = "seat_authority_unconfigured"
    runtime = None
    if reason is None:
        runtime = await asyncio.to_thread(_resolve_or_none)
        reason = unsupported_reason(runtime)
    return web.json_response({
        "contract": CONTRACT, "available": reason is None, "reason": reason,
        "api_mode": (runtime or {}).get("api_mode") if reason is None else None,
        # The route the product binds an admission to; never a credential or base URL.
        "seat_route": seat_route(runtime) if reason is None and runtime else None,
        "max_input_bytes": MAX_INPUT_BYTES, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_admission_ttl_seconds": MAX_ADMISSION_TTL_S, "tools": [], "invocations_per_admission": 1,
        "authority": "canonical_at_dispatch"})


def _http_routes(adapter: Any) -> list:
    return [
        ("GET", "/v1/seat/contract", functools.partial(handle_seat_contract, adapter)),
        ("POST", "/v1/seat/completions", functools.partial(handle_seat_completion, adapter)),
        ("GET", "/v1/seat/completions/{admission_key}", functools.partial(handle_seat_readback, adapter))]

