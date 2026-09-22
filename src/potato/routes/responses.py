"""OpenAI Responses API compatible route (POST /v1/responses).

Translates the Responses API surface to Chat Completions so every
OpenAI-compatible upstream provider works — not just OpenAI itself.
Mirrors the Claude/Anthropic route pattern in ``routes/claude.py``.

Endpoints:
- POST /v1/responses  (stream + non-stream)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from potato.routes.openai import _chat_like

logger = logging.getLogger(__name__)

router = APIRouter(tags=["responses"])


def _extract_text_from_input_item(item: Any) -> tuple[str, str, list[dict[str, Any]]]:
    """Return (role, text, multimodal_parts) from a Responses input item.

    Accepts Responses message items (``{role, content}`` where content is a
    string or a list of typed parts) and Chat-Compat message dicts (which are
    a strict subset of Responses input items — pass-through works).

    Multimodal parts (``input_image_url``, ``input_image``, ``input_audio``,
    ``input_video``) are returned as OpenAI Chat-content part dicts so the
    upstream provider receives native multimodal content instead of a
    text placeholder.
    """
    if not isinstance(item, dict):
        return "user", str(item or ""), []
    role = str(item.get("role") or "user")
    content = item.get("content")
    if isinstance(content, str):
        return role, content, []
    if isinstance(content, list):
        parts: list[str] = []
        multi: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict):
                # Responses input text part: {type: "input_text", text: "..."}
                # Responses output text part: {type: "output_text", text: "..."}
                # Chat-compat: {type: "text", text: "..."}
                t = block.get("type") or ""
                text = block.get("text")
                if isinstance(text, str) and t in ("", "text", "input_text", "output_text"):
                    parts.append(text)
                elif t in ("input_image_url", "input_image") and isinstance(block.get("image_url"), dict):
                    url = block["image_url"].get("url")
                    if isinstance(url, str):
                        multi.append({"type": "image_url", "image_url": {"url": url}})
                elif t == "input_audio" and isinstance(block.get("input_audio"), dict):
                    audio = block["input_audio"]
                    if isinstance(audio.get("data"), str):
                        multi.append({
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio["data"],
                                "format": (audio.get("format") or "wav").lower(),
                            },
                        })
                elif t in ("input_video", "video_url") and isinstance(block.get("video_url"), dict):
                    url = block["video_url"].get("url")
                    if isinstance(url, str):
                        multi.append({"type": "video_url", "video_url": {"url": url}})
            elif isinstance(block, str):
                parts.append(block)
        return role, "\n".join(parts), multi
    return role, str(content or ""), []


def transform_responses_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses API request body to a Chat Completions body.

    Spec: https://platform.openai.com/docs/api-reference/responses
    Key field mappings:
      - ``input`` (str | list[Item]) → ``messages``
      - ``instructions`` (str) → leading ``system`` message
      - ``max_output_tokens`` → ``max_tokens``
      - ``temperature``, ``top_p``, ``stream``, ``stop`` → passthrough
      - ``tools`` (internally-tagged) → ``tools`` (externally-tagged)
      - ``text.format`` (json_schema) → ``response_format``
      - ``reasoning.effort`` / ``reasoning_effort`` → ``reasoning_effort``
        (carried through so reasoning-capable upstreams honor client intent;
        the Chat path's inject_default_reasoning_effort won't override it)
    """
    out: dict[str, Any] = {}

    # Scalar passthroughs shared by both APIs.
    for key in (
        "model",
        "temperature",
        "top_p",
        "stream",
        "stop",
        "user",
        "seed",
        "presence_penalty",
        "frequency_penalty",
    ):
        if key in body and body[key] is not None:
            out[key] = body[key]

    # max_output_tokens → max_tokens
    if "max_output_tokens" in body and body["max_output_tokens"] is not None:
        out["max_tokens"] = body["max_output_tokens"]
    elif "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]

    # reasoning_effort: carry through to the Chat body so reasoning-capable
    # upstreams honor client intent. Accept both the top-level OpenAI field
    # and the nested ``reasoning: {effort: ...}`` Responses shape.
    reasoning_effort = body.get("reasoning_effort")
    if reasoning_effort is None and isinstance(body.get("reasoning"), dict):
        reasoning_effort = body["reasoning"].get("effort")
    if reasoning_effort is not None:
        out["reasoning_effort"] = reasoning_effort

    # Build messages from instructions + input.
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                continue
            itype = item.get("type") or ""
            if itype == "function_call_output":
                # Responses tool result item → Chat tool message.
                # Shape: {type, call_id, output}
                out_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or "",
                    "content": str(item.get("output") or ""),
                }
                messages.append(out_msg)
                continue
            if itype == "function_call":
                # An assistant's prior tool call, replayed as context.
                # Shape: {type, call_id, name, arguments}
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": item.get("call_id") or item.get("id") or "",
                                "type": "function",
                                "function": {
                                    "name": item.get("name") or "",
                                    "arguments": str(item.get("arguments") or ""),
                                },
                            }
                        ],
                    }
                )
                continue
            if itype in ("reasoning",):
                # Reasoning items don't carry chat text — skip.
                continue
            # Default: message-shaped item (has role + content).
            role, text, multi = _extract_text_from_input_item(item)
            if multi:
                # Multimodal user turn — emit OpenAI multi-part content
                # (text + image/audio/video parts) so vision-capable upstreams
                # receive native multimodal input instead of a text placeholder.
                content_parts: list[dict[str, Any]] = []
                if text:
                    content_parts.append({"type": "text", "text": text})
                content_parts.extend(multi)
                messages.append({"role": role, "content": content_parts})
            else:
                messages.append({"role": role, "content": text})

    out["messages"] = messages

    # Tools: Responses uses internally-tagged → Chat uses externally-tagged.
    raw_tools = body.get("tools")
    if isinstance(raw_tools, list) and raw_tools:
        chat_tools: list[dict[str, Any]] = []
        for tool in raw_tools:
            if not isinstance(tool, dict):
                continue
            ttype = tool.get("type") or ""
            if ttype == "function" and isinstance(tool.get("function"), dict):
                # Already Chat-shaped (externally-tagged) — pass through.
                chat_tools.append(tool)
                continue
            if ttype == "function" or "name" in tool:
                # Responses internally-tagged function tool:
                # {type:"function", name, description, parameters, strict?}
                chat_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name") or "",
                            "description": tool.get("description") or "",
                            "parameters": tool.get("parameters") or {},
                            **({"strict": tool["strict"]} if "strict" in tool else {}),
                        },
                    }
                )
                continue
            # Other built-in tool types (web_search, file_search, …) have no
            # Chat equivalent — drop them; the chat model can't honor them.
        if chat_tools:
            out["tools"] = chat_tools
        # tool_choice passes through unchanged between the two APIs.
        if body.get("tool_choice") is not None:
            out["tool_choice"] = body["tool_choice"]

    # Structured Outputs: text.format → response_format
    text_cfg = body.get("text")
    if isinstance(text_cfg, dict):
        fmt = text_cfg.get("format")
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            schema_cfg = fmt.get("json_schema") or fmt
            out["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_cfg.get("name") or "response",
                    **({"strict": schema_cfg["strict"]} if "strict" in schema_cfg else {}),
                    "schema": schema_cfg.get("schema") or schema_cfg,
                },
            }

    return out


def _now_iso() -> int:
    return int(time.time())


def _resp_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _msg_item_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _fc_item_id() -> str:
    return f"fc_{uuid.uuid4().hex}"


def transform_chat_to_responses_json(
    chat_resp: dict[str, Any], requested_model: str
) -> dict[str, Any]:
    """Convert a Chat Completions JSON response into a Responses JSON object."""
    rid = _resp_id()
    model = chat_resp.get("model") or requested_model or "potato/auto"
    choices = chat_resp.get("choices") or []
    usage = chat_resp.get("usage") or {}

    output_items: list[dict[str, Any]] = []
    output_text = ""
    if choices and isinstance(choices[0], dict):
        first = choices[0]
        msg = first.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            output_text = content
        elif isinstance(content, list):
            output_text = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        # Tool calls → function_call output items.
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            output_items.append(
                {
                    "id": _fc_item_id(),
                    "type": "function_call",
                    "call_id": tc.get("id") or _fc_item_id(),
                    "name": fn.get("name") or "",
                    "arguments": fn.get("arguments") or "",
                    "status": "completed",
                }
            )
        # Always emit a message item (even if empty) — clients expect it.
        output_items.append(
            {
                "id": _msg_item_id(),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                        "annotations": [],
                    }
                ],
            }
        )

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "id": rid,
        "object": "response",
        "created_at": _now_iso(),
        "model": model,
        "output": output_items,
        "output_text": output_text,
        "status": "completed",
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def transform_chat_sse_to_responses_sse(
    chat_stream: StreamingResponse, requested_model: str
) -> AsyncGenerator[bytes, None]:
    """Convert a Chat Completions SSE stream into a Responses SSE stream.

    Emits the minimal event set Cursor/SDKs require:
      response.created → response.output_item.added (message) →
      response.content_part.added → response.output_text.delta* →
      response.output_text.done → response.content_part.done →
      response.output_item.done → response.completed

    Tool-call deltas map to response.function_call_arguments.delta/done.
    """
    rid = _resp_id()
    model = requested_model or "potato/auto"
    created_at = _now_iso()

    def frame(event_type: str, payload: dict[str, Any]) -> bytes:
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()

    # response.created
    yield frame(
        "response.created",
        {
            "type": "response.created",
            "response": {
                "id": rid,
                "object": "response",
                "created_at": created_at,
                "model": model,
                "status": "in_progress",
                "output": [],
                "usage": None,
            },
        },
    )

    msg_item_id = _msg_item_id()
    msg_item = {
        "id": msg_item_id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }

    # response.output_item.added (the assistant message)
    yield frame(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": msg_item,
        },
    )

    # response.content_part.added (output_text part)
    yield frame(
        "response.content_part.added",
        {
            "type": "response.content_part.added",
            "item_id": msg_item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
    )

    accumulated_text: list[str] = []
    # Track in-flight tool calls by Chat tool_call id → Responses fc item.
    tool_items: dict[str, dict[str, Any]] = {}
    tool_arg_buffers: dict[str, str] = {}

    buffer = ""
    out_tokens = 0
    async for chunk in chat_stream.body_iterator:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8", errors="replace")
        else:
            buffer += str(chunk)

        lines = buffer.split("\n")
        buffer = lines.pop()  # keep incomplete tail
        for line in lines:
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                continue
            try:
                payload = json.loads(data_str)
            except Exception:
                continue

            if "error" in payload:
                err_obj = payload.get("error") or {}
                err_msg = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
                yield frame(
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": {
                            "id": rid,
                            "status": "failed",
                            "error": {"code": "upstream_error", "message": err_msg},
                        },
                    },
                )
                return

            choices = payload.get("choices") or []
            if not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta") or {}

            # Text deltas
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                accumulated_text.append(piece)
                out_tokens += 1
                yield frame(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": msg_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": piece,
                    },
                )

            # Tool call deltas
            tool_calls = delta.get("tool_calls") or []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id") or tc.get("index") or "0"
                if tc_id not in tool_items:
                    # First sight of this tool call → emit a new output item.
                    fc_item = {
                        "id": _fc_item_id(),
                        "type": "function_call",
                        "call_id": tc.get("id") or _fc_item_id(),
                        "name": (tc.get("function") or {}).get("name") or "",
                        "arguments": "",
                        "status": "in_progress",
                    }
                    tool_items[tc_id] = fc_item
                    tool_arg_buffers[tc_id] = ""
                    out_idx = len(tool_items)  # 1-based if after message
                    yield frame(
                        "response.output_item.added",
                        {
                            "type": "response.output_item.added",
                            "output_index": out_idx,
                            "item": fc_item,
                        },
                    )
                else:
                    fc_item = tool_items[tc_id]
                fn = tc.get("function") or {}
                arg_piece = fn.get("arguments") or ""
                if arg_piece:
                    tool_arg_buffers[tc_id] += arg_piece
                    yield frame(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": fc_item["id"],
                            "output_index": list(tool_items).index(tc_id),
                            "delta": arg_piece,
                        },
                    )

            finish = choice.get("finish_reason")
            if finish == "error":
                yield frame(
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": {
                            "id": rid,
                            "status": "failed",
                            "error": {
                                "code": "upstream_stream_error",
                                "message": "Upstream stream terminated with error",
                            },
                        },
                    },
                )
                return
            if finish:
                # Finalize any in-flight tool calls.
                for tc_id, fc_item in tool_items.items():
                    fc_item["arguments"] = tool_arg_buffers.get(tc_id, "")
                    fc_item["status"] = "completed"
                    out_idx = list(tool_items).index(tc_id)
                    yield frame(
                        "response.function_call_arguments.done",
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": fc_item["id"],
                            "output_index": out_idx,
                            "arguments": fc_item["arguments"],
                        },
                    )
                    yield frame(
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": out_idx,
                            "item": {**fc_item, "status": "completed"},
                        },
                    )

    full_text = "".join(accumulated_text)

    # response.output_text.done
    yield frame(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": msg_item_id,
            "output_index": 0,
            "content_index": 0,
            "text": full_text,
        },
    )

    # response.content_part.done
    yield frame(
        "response.content_part.done",
        {
            "type": "response.content_part.done",
            "item_id": msg_item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": full_text, "annotations": []},
        },
    )

    # response.output_item.done (message)
    final_msg = {
        "id": msg_item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": full_text, "annotations": []}],
    }
    yield frame(
        "response.output_item.done",
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": final_msg,
        },
    )

    # response.completed
    yield frame(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "id": rid,
                "object": "response",
                "created_at": created_at,
                "model": model,
                "status": "completed",
                "output": [final_msg, *tool_items.values()],
                "output_text": full_text,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": max(1, out_tokens),
                    "total_tokens": max(1, out_tokens),
                },
            },
        },
    )


async def _handle_responses(request: Request) -> JSONResponse | StreamingResponse:
    """Translate /v1/responses → /chat/completions upstream, translate back."""
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    requested_model = body.get("model", "potato/auto")
    is_stream = bool(body.get("stream"))

    chat_body = transform_responses_to_chat(body)
    # Preserve stream flag for the upstream chat call.
    chat_body["stream"] = is_stream

    body_sent = False

    async def _custom_receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {
                "type": "http.request",
                "body": json.dumps(chat_body).encode("utf-8"),
                "more_body": False,
            }
        return await request.receive()

    # x-api-key → Authorization (Anthropic SDK / some OpenAI SDK configs).
    scope = dict(request.scope)
    headers = list(scope.get("headers", []))
    x_api_key = request.headers.get("x-api-key") or request.headers.get("X-Api-Key")
    if x_api_key and not request.headers.get("authorization"):
        headers.append((b"authorization", f"Bearer {x_api_key}".encode()))
        scope["headers"] = headers

    req_clone = Request(scope, _custom_receive)
    resp = await _chat_like(req_clone, upstream_path="/chat/completions")

    # Non-streaming JSON translation back to Responses shape.
    if isinstance(resp, JSONResponse):
        try:
            resp_json = json.loads(bytes(resp.body).decode("utf-8"))
            responses_json = transform_chat_to_responses_json(resp_json, requested_model)
            return JSONResponse(content=responses_json, status_code=resp.status_code)
        except Exception as exc:
            logger.warning("chat→responses JSON transform failed: %s", exc)
            return resp

    # Streaming SSE translation back to Responses event stream.
    if isinstance(resp, StreamingResponse):
        return StreamingResponse(
            transform_chat_sse_to_responses_sse(resp, requested_model),
            status_code=resp.status_code,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return resp


@router.post("/v1/responses", response_model=None)
async def responses_api(request: Request) -> JSONResponse | StreamingResponse:
    return await _handle_responses(request)
