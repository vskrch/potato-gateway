"""Claude UI / Anthropic Messages API compatible routes.

Endpoints:
- POST /chat
- POST /v1/messages
- POST /messages
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from potato.routes.openai import _chat_like

logger = logging.getLogger(__name__)

router = APIRouter(tags=["claude"])


def _extract_raw_tool_calls_from_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Recover tool_use blocks and strip raw tool tokens from open-source model responses."""
    if not text:
        return text, []

    extracted_tools: list[dict[str, Any]] = []
    clean_text = text

    # Pattern 1: <|tool_call:name{args}|>
    for m in re.finditer(r"<\|tool_call:([a-zA-Z0-9_-]+)(\{.*?\})\|>", text, re.DOTALL):
        name, args_str = m.group(1), m.group(2)
        try:
            input_data = json.loads(args_str)
        except Exception:
            input_data = {}
        extracted_tools.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:12]}",
            "name": name,
            "input": input_data,
        })
        clean_text = clean_text.replace(m.group(0), "")

    # Pattern 2: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            payload = json.loads(m.group(1))
            name = payload.get("name") or payload.get("function", {}).get("name", "tool")
            args = payload.get("arguments") or payload.get("parameters") or payload.get("input") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            extracted_tools.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:12]}",
                "name": str(name),
                "input": args if isinstance(args, dict) else {},
            })
            clean_text = clean_text.replace(m.group(0), "")
        except Exception:
            pass

    # Strip raw section markers
    clean_text = re.sub(r"<\|tool_calls_section_begin\|>", "", clean_text)
    clean_text = re.sub(r"<\|tool_calls_section_end\|>", "", clean_text)
    clean_text = clean_text.strip()

    return clean_text, extracted_tools


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(str(block["text"]))
                elif btype == "tool_use":
                    name = block.get("name", "tool")
                    inp = json.dumps(block.get("input", {}))
                    parts.append(f"[Tool Call: {name}({inp})]")
                elif btype == "tool_result":
                    tool_id = block.get("tool_use_id", "")
                    sub_content = _extract_text_content(block.get("content"))
                    parts.append(f"[Tool Result {tool_id}: {sub_content}]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content or "")


def is_anthropic_request(body: dict[str, Any], path: str) -> bool:
    """Return True if request matches Anthropic Messages API format or endpoint."""
    if "messages" in path or "system" in body or "anthropic-version" in str(body):
        return True
    msgs = body.get("messages", [])
    if isinstance(msgs, list) and msgs:
        first = msgs[0]
        if isinstance(first, dict) and isinstance(first.get("content"), list):
            return True
    return False


def transform_anthropic_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Convert Anthropic Messages API payload to OpenAI Chat Completions payload."""
    openai_body: dict[str, Any] = {}

    # Copy parameters
    for key in ("temperature", "top_p", "stream", "stop"):
        if key in body:
            openai_body[key] = body[key]

    if "max_tokens" in body:
        openai_body["max_tokens"] = body["max_tokens"]

    # Model default — Claude Code requests always route through the
    # coding intent chain (potato/auto-coding) which selects the best
    # available coding model from the live catalog.
    raw_model = str(body.get("model") or "").strip()
    if not raw_model or raw_model.startswith("claude") or raw_model == "auto":
        openai_body["model"] = "potato/auto-coding"
    else:
        openai_body["model"] = raw_model

    # Transform Anthropic tools format to OpenAI tools format
    if "tools" in body and isinstance(body["tools"], list):
        openai_tools = []
        for tool in body["tools"]:
            if isinstance(tool, dict):
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                    }
                })
        openai_body["tools"] = openai_tools

    # Transform Anthropic tool_choice format to OpenAI tool_choice format
    if "tool_choice" in body:
        tc = body["tool_choice"]
        if isinstance(tc, str):
            openai_body["tool_choice"] = tc
        elif isinstance(tc, dict):
            ttype = tc.get("type")
            if ttype == "auto":
                openai_body["tool_choice"] = "auto"
            elif ttype == "any":
                openai_body["tool_choice"] = "required"
            elif ttype == "tool":
                tname = tc.get("name")
                if tname:
                    openai_body["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tname},
                    }

    # Construct messages array
    openai_msgs: list[dict[str, Any]] = []

    # Handle system prompt
    system_prompt = body.get("system")
    if system_prompt:
        sys_text = _extract_text_content(system_prompt)
        if sys_text:
            openai_msgs.append({"role": "system", "content": sys_text})

    # Convert Anthropic messages
    anthropic_msgs = body.get("messages", [])
    if isinstance(anthropic_msgs, list):
        for msg in anthropic_msgs:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content_raw = msg.get("content")

            if isinstance(content_raw, str):
                openai_msgs.append({"role": role, "content": content_raw})
            elif isinstance(content_raw, list):
                text_parts = []
                image_blocks = []
                audio_blocks = []
                video_blocks = []
                tool_calls = []
                tool_results = []

                for block in content_raw:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "image":
                        source = block.get("source", {})
                        if isinstance(source, dict):
                            stype = source.get("type")
                            if stype == "base64":
                                mtype = source.get("media_type", "image/jpeg")
                                b64 = source.get("data", "")
                                image_blocks.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{mtype};base64,{b64}"},
                                })
                            elif stype == "url":
                                image_blocks.append({
                                    "type": "image_url",
                                    "image_url": {"url": source.get("url", "")},
                                })
                    elif btype == "audio":
                        # Anthropic audio block: {type:"audio", source:{type:"base64"|"url", media_type, data|url}}
                        source = block.get("source", {})
                        if isinstance(source, dict):
                            stype = source.get("type")
                            mtype = source.get("media_type", "audio/wav")
                            if stype == "base64":
                                b64 = source.get("data", "")
                                # ponytail: OpenAI input_audio uses inline base64 data URL.
                                # No URL form in the OpenAI spec yet; wrap as data URL.
                                audio_blocks.append({
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": b64,
                                        "format": (mtype.split("/")[-1] or "wav").lower(),
                                    },
                                })
                            elif stype == "url":
                                # OpenAI has no native audio URL part; embed as data URL
                                # when upstream supports it, else fall through to text note.
                                audio_blocks.append({
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": source.get("url", ""),
                                        "format": "url",
                                    },
                                })
                    elif btype == "video":
                        # Anthropic has no official video block yet, but Claude Code / custom
                        # clients may send one. Treat as OpenAI video_url (experimental).
                        source = block.get("source", {})
                        if isinstance(source, dict):
                            stype = source.get("type")
                            mtype = source.get("media_type", "video/mp4")
                            if stype == "base64":
                                b64 = source.get("data", "")
                                video_blocks.append({
                                    "type": "video_url",
                                    "video_url": {"url": f"data:{mtype};base64,{b64}"},
                                })
                            elif stype == "url":
                                video_blocks.append({
                                    "type": "video_url",
                                    "video_url": {"url": source.get("url", "")},
                                })
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                    elif btype == "tool_result":
                        res_text = _extract_text_content(block.get("content"))
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": res_text,
                        })

                if role == "assistant":
                    asst_msg: dict[str, Any] = {"role": "assistant"}
                    if text_parts:
                        asst_msg["content"] = "\n".join(text_parts)
                    if tool_calls:
                        asst_msg["tool_calls"] = tool_calls
                    openai_msgs.append(asst_msg)
                else:  # user role
                    if image_blocks or audio_blocks or video_blocks:
                        multi_content: list[dict[str, Any]] = []
                        if text_parts:
                            multi_content.append({"type": "text", "text": "\n".join(text_parts)})
                        multi_content.extend(image_blocks)
                        multi_content.extend(audio_blocks)
                        multi_content.extend(video_blocks)
                        openai_msgs.append({"role": "user", "content": multi_content})
                    elif text_parts:
                        openai_msgs.append({"role": "user", "content": "\n".join(text_parts)})
                    for tr in tool_results:
                        openai_msgs.append(tr)
            else:
                openai_msgs.append({"role": role, "content": str(content_raw or "")})

    openai_body["messages"] = openai_msgs
    return openai_body


def transform_openai_to_anthropic_json(
    openai_resp: dict[str, Any], requested_model: str
) -> dict[str, Any]:
    """Convert OpenAI Chat Completion JSON response into Anthropic Message JSON response."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    choices = openai_resp.get("choices", [])
    text_content = ""
    stop_reason = "end_turn"
    content_blocks: list[dict[str, Any]] = []

    if choices and isinstance(choices, list):
        first = choices[0]
        if isinstance(first, dict):
            msg_obj = first.get("message", {})
            text_content = msg_obj.get("content", "") or ""
            if text_content:
                content_blocks.append({"type": "text", "text": text_content})

            tool_calls = msg_obj.get("tool_calls", [])
            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        args_raw = fn.get("arguments", "{}")
                        try:
                            parsed_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        except Exception:
                            parsed_args = {}
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                            "name": fn.get("name", "tool"),
                            "input": parsed_args,
                        })

                if content_blocks:
                    stop_reason = "tool_use"
            elif text_content:
                # Auto-recovery: recover tool calls printed as raw tokens/XML by open-source models
                clean_text, recovered = _extract_raw_tool_calls_from_text(text_content)
                if recovered:
                    content_blocks = []
                    if clean_text:
                        content_blocks.append({"type": "text", "text": clean_text})
                    content_blocks.extend(recovered)
                    stop_reason = "tool_use"

            reason = first.get("finish_reason")
            if reason == "length":
                stop_reason = "max_tokens"
            elif reason == "stop" and not tool_calls and stop_reason != "tool_use":
                stop_reason = "end_turn"

    if not content_blocks:
        content_blocks.append({"type": "text", "text": text_content})

    usage = openai_resp.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    model_name = openai_resp.get("model") or requested_model or "potato/auto"

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    }


async def transform_openai_to_anthropic_sse_stream(
    openai_stream: StreamingResponse, requested_model: str
) -> AsyncGenerator[bytes, None]:
    """Convert OpenAI SSE stream chunks into Anthropic SSE event frames."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    model_name = requested_model or "potato/auto"

    # Emit message_start
    msg_start_event = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(msg_start_event)}\n\n".encode()

    text_block_started = False
    active_tool_calls: dict[int, dict[str, Any]] = {}
    next_block_index = 0
    out_tokens = 0
    buffer = ""
    has_tool_calls = False

    has_error = False

    async for chunk in openai_stream.body_iterator:
        if isinstance(chunk, bytes):
            buffer += chunk.decode("utf-8", errors="replace")
        else:
            buffer += str(chunk)

        lines = buffer.split("\n")
        buffer = lines.pop()  # Keep incomplete tail line in buffer

        for line in lines:
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    continue
                try:
                    payload = json.loads(data_str)
                    if "error" in payload:
                        err_obj = payload.get("error") or {}
                        err_msg = (
                            err_obj.get("message")
                            if isinstance(err_obj, dict)
                            else str(err_obj)
                        )
                        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': err_msg}})}\n\n".encode()
                        has_error = True
                        break
                    choices = payload.get("choices", [])
                    if choices and isinstance(choices, list):
                        delta = choices[0].get("delta", {})
                        content_piece = delta.get("content")
                        if content_piece:
                            if not text_block_started:
                                block_start = {
                                    "type": "content_block_start",
                                    "index": next_block_index,
                                    "content_block": {"type": "text", "text": ""},
                                }
                                yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n".encode()
                                text_block_started = True
                                next_block_index += 1

                            out_tokens += 1
                            delta_event = {
                                "type": "content_block_delta",
                                "index": 0 if text_block_started else next_block_index - 1,
                                "delta": {"type": "text_delta", "text": content_piece},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n".encode()

                        # Tool calls streaming support
                        tc_deltas = delta.get("tool_calls")
                        if tc_deltas and isinstance(tc_deltas, list):
                            for tc in tc_deltas:
                                idx = tc.get("index", 0)
                                fn = tc.get("function", {})
                                if idx not in active_tool_calls:
                                    t_id = tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"
                                    t_name = fn.get("name", "tool")
                                    block_idx = next_block_index
                                    next_block_index += 1
                                    active_tool_calls[idx] = {
                                        "block_index": block_idx,
                                        "id": t_id,
                                        "name": t_name,
                                    }
                                    has_tool_calls = True
                                    tool_start = {
                                        "type": "content_block_start",
                                        "index": block_idx,
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": t_id,
                                            "name": t_name,
                                            "input": {},
                                        },
                                    }
                                    yield f"event: content_block_start\ndata: {json.dumps(tool_start)}\n\n".encode()

                                args_delta = fn.get("arguments")
                                if args_delta:
                                    tool_info = active_tool_calls[idx]
                                    out_tokens += 1
                                    json_delta = {
                                        "type": "content_block_delta",
                                        "index": tool_info["block_index"],
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": args_delta,
                                        },
                                    }
                                    yield f"event: content_block_delta\ndata: {json.dumps(json_delta)}\n\n".encode()
                except Exception:
                    pass

    if not has_error:
        # Stop all open content blocks
        if text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n".encode()

        for tc_info in active_tool_calls.values():
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': tc_info['block_index']})}\n\n".encode()

        stop_reason = "tool_use" if has_tool_calls else "end_turn"
        msg_delta_event = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": max(1, out_tokens)},
        }
        yield f"event: message_delta\ndata: {json.dumps(msg_delta_event)}\n\n".encode()

        msg_stop_event = {"type": "message_stop"}
        yield f"event: message_stop\ndata: {json.dumps(msg_stop_event)}\n\n".encode()


async def _handle_claude_or_chat(request: Request) -> JSONResponse | StreamingResponse:
    """Unified handler for /chat, /v1/messages, and /messages."""
    path = request.url.path
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    anthropic_format = is_anthropic_request(body, path)
    requested_model = body.get("model", "potato/auto")

    if anthropic_format:
        openai_payload = transform_anthropic_to_openai(body)
    else:
        openai_payload = body
        if not openai_payload.get("model"):
            openai_payload["model"] = "potato/auto-coding"

    body_sent = False

    async def _custom_receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {
                "type": "http.request",
                "body": json.dumps(openai_payload).encode("utf-8"),
                "more_body": False,
            }
        return await request.receive()

    # Support x-api-key header for Claude Code CLI and Anthropic SDKs
    scope = dict(request.scope)
    headers = list(scope.get("headers", []))
    x_api_key = request.headers.get("x-api-key") or request.headers.get("X-Api-Key")
    if x_api_key and not request.headers.get("authorization"):
        headers.append((b"authorization", f"Bearer {x_api_key}".encode()))
        scope["headers"] = headers

    # Replace receive method on request
    req_clone = Request(scope, _custom_receive)

    resp = await _chat_like(req_clone, upstream_path="/chat/completions")

    if not anthropic_format:
        return resp

    if isinstance(resp, JSONResponse):
        try:
            resp_bytes_raw = bytes(resp.body)
            resp_json = json.loads(resp_bytes_raw.decode("utf-8"))
            if resp.status_code >= 400:
                err_obj = resp_json.get("error", {}) if isinstance(resp_json, dict) else {}
                err_msg = err_obj.get("message") if isinstance(err_obj, dict) else str(resp_json)
                err_type = "invalid_request_error" if resp.status_code < 500 else "api_error"
                return JSONResponse(
                    content={
                        "type": "error",
                        "error": {
                            "type": err_type,
                            "message": err_msg or "API Request Failed"
                        }
                    },
                    status_code=resp.status_code
                )
            anthropic_json = transform_openai_to_anthropic_json(resp_json, requested_model)
            return JSONResponse(content=anthropic_json, status_code=resp.status_code)
        except Exception as exc:
            logger.warning("Failed to convert OpenAI JSON to Anthropic JSON: %s", exc)
            return resp

    if isinstance(resp, StreamingResponse):
        return StreamingResponse(
            transform_openai_to_anthropic_sse_stream(resp, requested_model),
            status_code=resp.status_code,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return resp


@router.post("/chat", response_model=None)
async def chat_endpoint(request: Request) -> JSONResponse | StreamingResponse:
    return await _handle_claude_or_chat(request)


@router.post("/v1/messages", response_model=None)
async def v1_messages_endpoint(request: Request) -> JSONResponse | StreamingResponse:
    return await _handle_claude_or_chat(request)


@router.post("/messages", response_model=None)
async def messages_endpoint(request: Request) -> JSONResponse | StreamingResponse:
    return await _handle_claude_or_chat(request)
