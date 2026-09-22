"""OpenAI / Cursor client compatibility helpers.

Normalize upstream SSE and JSON payloads so clients always see standard
OpenAI fields. Reasoning (thinking) is kept in ``reasoning_content`` and
**never** mirrored into ``content`` — the thinking phase emits
``content: null`` with ``reasoning_content`` populated, and the answer
phase emits ``content`` with ``reasoning_content: null``. This keeps
AI SDK / Kilo from double-rendering private thinking as both reasoning
blocks and visible assistant text. The reasoning field is canonicalized
to ``reasoning_content`` regardless of which upstream field name was used
(``reasoning`` vs ``reasoning_content``).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)

# Fields some upstreams reject. OpenRouter client-only fields are stripped by
# strip_router_client_fields AFTER parse_auto_router_options — not here.
# prompt_cache_key / user are forwarded (providers ignore unknown fields).
_STRIP_BODY_KEYS = {
    "service_tier",
    "safety_identifier",
    "store",
    "metadata",
}


def openai_error(
    message: str,
    *,
    code: str | None = None,
    type_: str = "invalid_request_error",
    param: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """OpenAI-compatible error envelope (top-level ``error`` object)."""
    err: dict[str, Any] = {
        "message": message,
        "type": type_,
        "code": code,
        "param": param,
    }
    if metadata:
        err["metadata"] = metadata
    return {"error": err}


def wrap_upstream_error(body: Any, *, status: int = 502) -> dict[str, Any]:
    """Normalize non-OpenAI upstream error bodies into the OpenAI envelope."""
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return body
    raw: Any = body
    if isinstance(body, dict) and "detail" in body and "error" not in body:
        raw = body.get("detail")
    return openai_error(
        str(raw)[:2000] if raw is not None else f"Upstream error HTTP {status}",
        code="upstream_error",
        type_="server_error",
        metadata={"raw": raw if not isinstance(raw, str) or len(raw) < 500 else raw[:500]},
    )


def inject_system_prompt(body: dict[str, Any], prompt: str | None) -> dict[str, Any]:
    """Prepend a universal system prompt to ``body["messages"]``.

    No-op when ``prompt`` is empty. When the first message is already a
    ``system`` message with string content, the prompt is merged into the
    front of that content (keeps a single system turn). Otherwise a new
    ``system`` message is inserted at index 0. Multimodal system content
    (list of parts) is left untouched and a new system message is inserted
    before it — providers honor multiple system messages in order.
    """
    if not prompt:
        return body
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        # Nothing to attach to; let the caller's body flow as-is rather than
        # fabricating a messages array for endpoints that don't use it.
        return body
    first = msgs[0]
    if (
        isinstance(first, dict)
        and first.get("role") == "system"
        and isinstance(first.get("content"), str)
    ):
        merged = {**first, "content": f"{prompt}\n\n{first['content']}"}
        body = {**body, "messages": [merged, *msgs[1:]]}
    else:
        body = {**body, "messages": [{"role": "system", "content": prompt}, *msgs]}
    return body


def normalize_reasoning_effort(
    body: dict[str, Any],
    *,
    routed_model: str | None,
    registry: Any | None,
    default_effort: str = "",
) -> dict[str, Any]:
    """Normalize ``reasoning_effort`` for one routed model.

    Called per-model in the fallback chain so a reasoning head failing over to
    a non-reasoning model strips the field instead of 400ing the upstream.

    - Client set an explicit value: honor it for reasoning-capable models;
      strip it for non-reasoning models (most upstreams 400 on an unknown
      field, and a non-reasoning model has no thinking to tune).
    - Client did not set one: inject ``default_effort`` for reasoning-capable
      models when ``default_effort`` is non-empty; leave non-reasoning models
      untouched.
    - Unknown model (not in capabilities): pass through unchanged — we don't
      guess, preserving forward-compatibility with newly-added models.

    Capability is read from the ladder ``capabilities`` dict
    (``supports_reasoning``). No registry / no model → pass through.
    """
    if routed_model is None or registry is None:
        return body
    caps = getattr(getattr(registry, "ladder", None), "capabilities", None)
    if not caps:
        return body
    flags = caps.get(routed_model)
    if flags is None:
        return body  # unknown model — forward-compatible passthrough
    supports_reasoning = flags.get("supports_reasoning") is True
    has_explicit = "reasoning_effort" in body
    if supports_reasoning:
        if not has_explicit and default_effort:
            return {**body, "reasoning_effort": default_effort}
        return body  # honor client's explicit value as-is
    # Non-reasoning model: strip any reasoning_effort so the upstream doesn't
    # 400 on a field it doesn't understand.
    if has_explicit:
        body = {k: v for k, v in body.items() if k != "reasoning_effort"}
    return body


def sanitize_chat_body(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize client request for OpenAI-compatible upstreams (Cursor-safe).

    Raises ValueError with code ``n_not_supported`` when ``n > 1``.
    """
    out = dict(body)

    # max_completion_tokens (newer OpenAI / Cursor) → max_tokens
    if "max_tokens" not in out and out.get("max_completion_tokens") is not None:
        out["max_tokens"] = out.pop("max_completion_tokens")
    else:
        out.pop("max_completion_tokens", None)

    # stream_options is OpenAI-only; keep if stream else drop
    if not out.get("stream"):
        out.pop("stream_options", None)

    for k in list(_STRIP_BODY_KEYS):
        out.pop(k, None)

    n = out.get("n")
    if n not in (None, 1):
        raise ValueError("n_not_supported")

    # Filter invalid / None messages from messages array
    msgs = out.get("messages")
    if msgs is not None:
        if isinstance(msgs, list):
            out["messages"] = [m for m in msgs if isinstance(m, dict)]
        else:
            out["messages"] = []

    # max_tokens normalization: cast numeric strings, drop non-positive values
    mt = out.get("max_tokens")
    if isinstance(mt, str) and mt.isdigit():
        out["max_tokens"] = int(mt)
    elif isinstance(mt, (int, float)) and mt <= 0:
        out.pop("max_tokens", None)

    # temperature normalization: cast string, clamp negative to 0.0
    temp = out.get("temperature")
    if isinstance(temp, str):
        try:
            out["temperature"] = float(temp)
        except ValueError:
            out.pop("temperature", None)
    if isinstance(out.get("temperature"), (int, float)) and out["temperature"] < 0:
        out["temperature"] = 0.0

    # top_p normalization: cast string, clamp to [0.0, 1.0]
    top_p = out.get("top_p")
    if isinstance(top_p, str):
        try:
            out["top_p"] = float(top_p)
        except ValueError:
            out.pop("top_p", None)
    if isinstance(out.get("top_p"), (int, float)):
        out["top_p"] = max(0.0, min(1.0, float(out["top_p"])))

    # Empty tools → drop (some providers 400)
    tools = out.get("tools")
    if tools is not None and not tools:
        out.pop("tools", None)
        if out.get("tool_choice") in (None, "auto", "none"):
            out.pop("tool_choice", None)

    return out


def normalize_message_dict(msg: dict[str, Any]) -> dict[str, Any]:
    """Normalize a non-streaming assistant message for OpenAI clients.

    RC-1/RC-2 fixes:
    - Never mirror ``reasoning_content`` into ``content``. The answer lives
      in ``content``; private thinking lives in ``reasoning_content``. Mixing
      them causes AI SDK / Kilo to render thinking as visible assistant text.
    - Canonicalize the reasoning field to ``reasoning_content`` regardless of
      which upstream field name was used.
    """
    if not isinstance(msg, dict):
        return msg
    msg = dict(msg)
    # RC-2: canonicalize reasoning field name → always reasoning_content
    reasoning_alt = msg.pop("reasoning", None)
    if reasoning_alt and not msg.get("reasoning_content"):
        msg["reasoning_content"] = reasoning_alt
    # RC-1: some upstreams (nvidia/nemotron) mirror thinking into both
    # content and reasoning_content. Drop the duplicate from content so
    # clients never see thinking as visible assistant text (RCA §5: the
    # two fields must never be equal). Tools, if present, are preserved.
    rc = msg.get("reasoning_content")
    if (
        isinstance(rc, str)
        and rc
        and not (msg.get("tool_calls") or msg.get("function_call"))
        and msg.get("content") == rc
    ):
        msg["content"] = ""
    # RC-1: do not mirror reasoning into content. Keep them separated.
    return msg


def normalize_completion_json(body: Any, *, routed_model: str | None = None) -> Any:
    """Rewrite non-stream chat.completion JSON for OpenAI clients."""
    if not isinstance(body, dict):
        return body
    out = dict(body)
    if routed_model:
        out["model"] = routed_model
    choices = out.get("choices")
    if isinstance(choices, list):
        new_choices = []
        for ch in choices:
            if not isinstance(ch, dict):
                new_choices.append(ch)
                continue
            ch2 = dict(ch)
            msg = ch2.get("message")
            if isinstance(msg, dict):
                ch2["message"] = normalize_message_dict(dict(msg))
            # text completions style
            if ch2.get("text") in (None, "") and isinstance(ch2.get("message"), dict):
                pass
            new_choices.append(ch2)
        out["choices"] = new_choices
    return out


def _normalize_delta(delta: dict[str, Any]) -> dict[str, Any]:
    """Normalize one SSE delta for OpenAI-compatible clients.

    RC-1/RC-2/RC-3 fixes:
    - Never mirror reasoning into ``content``. The thinking phase emits
      ``content: null`` with ``reasoning_content`` populated; the answer
      phase emits ``content`` with ``reasoning_content: null``. This keeps
      the two phases cleanly segregated so AI SDK / Kilo does not double-
      render thinking as both reasoning blocks and visible content.
    - Canonicalize the reasoning field to ``reasoning_content`` regardless
      of which upstream field name was used (``reasoning`` vs
      ``reasoning_content``).
    """
    d = dict(delta)
    # RC-2: canonicalize reasoning field name → always reasoning_content
    reasoning_alt = d.pop("reasoning", None)
    if reasoning_alt and not d.get("reasoning_content"):
        d["reasoning_content"] = reasoning_alt
    # RC-1: some upstreams (nvidia/nemotron) stream thinking into both
    # content and reasoning_content with identical text. Drop the
    # duplicate from content so clients don't double-render thinking
    # as visible assistant text (RCA §5: the two fields must never be
    # equal). Tools, if present, are preserved.
    rc = d.get("reasoning_content")
    if (
        isinstance(rc, str)
        and rc
        and not (d.get("tool_calls") or d.get("function_call"))
        and d.get("content") == rc
    ):
        d["content"] = None
    # RC-1/RC-3: do NOT mirror reasoning into content. Keep phases segregated.
    # Ensure role on first useful delta (Cursor OpenAI client)
    if (
        d.get("content")
        or d.get("tool_calls")
        or d.get("function_call")
        or d.get("reasoning_content")
    ) and "role" not in d:
        d["role"] = "assistant"
    # tool_calls must stay intact (Cursor agent mode)
    return d


def normalize_sse_chunk_json(
    data: dict[str, Any], *, routed_model: str | None = None
) -> dict[str, Any]:
    out = dict(data)
    if routed_model:
        out["model"] = routed_model
    choices = out.get("choices")
    if isinstance(choices, list):
        new_ch = []
        for ch in choices:
            if not isinstance(ch, dict):
                new_ch.append(ch)
                continue
            ch2 = dict(ch)
            delta = ch2.get("delta")
            if isinstance(delta, dict):
                ch2["delta"] = _normalize_delta(delta)
            msg = ch2.get("message")
            if isinstance(msg, dict):
                ch2["message"] = normalize_message_dict(dict(msg))
            new_ch.append(ch2)
        out["choices"] = new_ch
    return out


_DATA_RE = re.compile(rb"^(data:\s*)(.*)$", re.I)


def transform_sse_bytes(chunk: bytes, *, routed_model: str | None = None) -> bytes:
    """
    Transform one or more SSE lines in a raw chunk.
    Safe to call on partial buffers only if caller splits by lines first.
    """
    if not chunk or chunk.strip() in (b"[DONE]", b"data: [DONE]"):
        return chunk
    # Fast path: no reasoning — only rewrite model via cheap substitution
    if b"reasoning" not in chunk:
        if routed_model is None or b'"model"' not in chunk:
            return chunk
        # Rewrite "model":"..." without full JSON parse
        return _rewrite_model_bytes(chunk, routed_model)

    lines = chunk.split(b"\n")
    out_lines: list[bytes] = []
    for line in lines:
        if not line.startswith(b"data:"):
            out_lines.append(line)
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]" or not payload:
            out_lines.append(line)
            continue
        try:
            obj = json.loads(payload.decode("utf-8", errors="replace"))
        except Exception:
            out_lines.append(line)
            continue
        if isinstance(obj, dict):
            obj = normalize_sse_chunk_json(obj, routed_model=routed_model)
            new_payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            out_lines.append(b"data: " + new_payload.encode("utf-8"))
        else:
            out_lines.append(line)
    # Preserve trailing newline behavior
    joined = b"\n".join(out_lines)
    if chunk.endswith(b"\n") and not joined.endswith(b"\n"):
        joined += b"\n"
    return joined


_MODEL_FIELD_RE = re.compile(rb'"model"\s*:\s*"(?:\\.|[^"\\])*"')


def _rewrite_model_bytes(chunk: bytes, routed_model: str) -> bytes:
    replacement = b'"model":' + json.dumps(routed_model, ensure_ascii=False).encode("utf-8")
    # Use lambda to avoid re.sub interpreting backslash escapes in the replacement
    return _MODEL_FIELD_RE.sub(lambda _: replacement, chunk, count=1)


async def normalize_sse_stream(
    source: AsyncIterator[bytes],
    *,
    routed_model: str | None = None,
) -> AsyncIterator[bytes]:
    """Line-buffer SSE stream and normalize each data: JSON event for Cursor."""
    from contextlib import suppress

    buffer = b""
    try:
        async for raw in source:
            buffer += raw
            while True:
                nl = buffer.find(b"\n")
                if nl < 0:
                    break
                line = buffer[: nl + 1]
                buffer = buffer[nl + 1 :]
                yield transform_sse_bytes(line, routed_model=routed_model)
        if buffer:
            yield transform_sse_bytes(buffer, routed_model=routed_model)
    finally:
        # D6b: deterministic cleanup — never rely on GC finalizers to release
        # the upstream iterator (and its key in_flight slot).
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            with suppress(Exception):
                await aclose()


def frame_sse_error(
    message: str,
    *,
    code: str = "upstream_error",
    status: int = 502,
    retry_after: str | None = None,
) -> bytes:
    """SSE-framed OpenAI error for StreamingResponse clients."""
    meta = {"retry_after": retry_after} if retry_after else None
    payload = openai_error(
        message,
        code=code,
        type_="server_error" if status >= 500 or status == 429 else "invalid_request_error",
        metadata=meta,
    )
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\ndata: [DONE]\n\n"


def json_body_to_sse(body: Any, *, routed_model: str | None = None) -> bytes:
    """Convert a non-streaming JSON completion into a one-shot SSE sequence (F-18)."""
    if not isinstance(body, dict):
        try:
            body = json.loads(body) if isinstance(body, (bytes, str)) else {"raw": body}
        except Exception:
            body = {"error": {"message": "upstream returned non-JSON body"}}
    if routed_model:
        body = {**body, "model": routed_model}
    choices = body.get("choices")
    chunks: list[bytes] = []
    if isinstance(choices, list) and choices:
        ch0 = choices[0] if isinstance(choices[0], dict) else {}
        msg = ch0.get("message") if isinstance(ch0.get("message"), dict) else {}
        delta: dict[str, Any] = {}
        if isinstance(msg, dict):
            if msg.get("content") is not None:
                delta["content"] = msg.get("content")
            # RC-2: forward reasoning_content (canonicalized) in JSON→SSE path
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if reasoning:
                delta["reasoning_content"] = reasoning
            if msg.get("tool_calls") is not None:
                delta["tool_calls"] = msg.get("tool_calls")
            if msg.get("role"):
                delta["role"] = msg.get("role")
        elif ch0.get("text") is not None:
            delta["content"] = ch0.get("text")
        chunk = {
            "id": body.get("id") or "potato-json-stream",
            "object": "chat.completion.chunk",
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": ch0.get("finish_reason") or "stop",
                }
            ],
        }
        if isinstance(body.get("usage"), dict):
            chunk["usage"] = body["usage"]
        chunks.append(b"data: " + json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\n\n")
    else:
        chunks.append(b"data: " + json.dumps(body, ensure_ascii=False).encode("utf-8") + b"\n\n")
    chunks.append(b"data: [DONE]\n\n")
    return b"".join(chunks)
