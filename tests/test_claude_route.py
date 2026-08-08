"""Unit tests for Claude UI / Anthropic Messages API compatible routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from potato.main import create_app
from potato.routes.claude import (
    is_anthropic_request,
    transform_anthropic_to_openai,
    transform_openai_to_anthropic_json,
)


def test_is_anthropic_request():
    assert is_anthropic_request({"system": "Hello"}, "/chat") is True
    assert is_anthropic_request({}, "/v1/messages") is True
    assert (
        is_anthropic_request(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}, "/chat"
        )
        is True
    )
    assert is_anthropic_request({"messages": [{"role": "user", "content": "hi"}]}, "/chat") is False


def test_transform_anthropic_to_openai():
    anthropic_payload = {
        "model": "claude-3-5-sonnet-20241022",
        "system": "Be concise",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Write a python function"}]}
        ],
        "max_tokens": 500,
        "temperature": 0.5,
        "tool_choice": {"type": "any"},
    }
    openai_payload = transform_anthropic_to_openai(anthropic_payload)
    # Claude Code requests are intentionally routed through the coding intent
    # chain (potato/auto-coding) — see routes/claude.py.
    assert openai_payload["model"] == "potato/auto-coding"
    assert openai_payload["max_tokens"] == 500
    assert openai_payload["tool_choice"] == "required"
    assert openai_payload["temperature"] == 0.5
    assert len(openai_payload["messages"]) == 2
    assert openai_payload["messages"][0] == {"role": "system", "content": "Be concise"}
    assert openai_payload["messages"][1] == {"role": "user", "content": "Write a python function"}


def test_transform_openai_to_anthropic_json():
    openai_resp = {
        "id": "chatcmpl-123",
        "model": "qwen/qwen3.5-122b-a10b",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "def foo(): pass"},
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 8},
    }
    anthropic_resp = transform_openai_to_anthropic_json(openai_resp, "potato/auto")
    assert anthropic_resp["type"] == "message"
    assert anthropic_resp["role"] == "assistant"
    assert anthropic_resp["content"][0]["text"] == "def foo(): pass"
    assert anthropic_resp["usage"]["input_tokens"] == 15
    assert anthropic_resp["usage"]["output_tokens"] == 8


def test_transform_anthropic_audio_block_to_openai():
    """Anthropic audio block (base64) → OpenAI input_audio part."""
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe this"},
                    {
                        "type": "audio",
                        "source": {
                            "type": "base64",
                            "media_type": "audio/wav",
                            "data": "UklGRiQAAABXQVZFZm10",
                        },
                    },
                ],
            }
        ],
    }
    out = transform_anthropic_to_openai(payload)
    user_msg = out["messages"][0]
    assert user_msg["role"] == "user"
    parts = user_msg["content"]
    assert isinstance(parts, list)
    audio_part = next(p for p in parts if p.get("type") == "input_audio")
    assert audio_part["input_audio"]["format"] == "wav"
    assert audio_part["input_audio"]["data"] == "UklGRiQAAABXQVZFZm10"


def test_transform_anthropic_video_block_to_openai():
    """Anthropic video block (url) → OpenAI video_url part."""
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this clip"},
                    {
                        "type": "video",
                        "source": {"type": "url", "url": "https://x/clip.mp4"},
                    },
                ],
            }
        ],
    }
    out = transform_anthropic_to_openai(payload)
    parts = out["messages"][0]["content"]
    video_part = next(p for p in parts if p.get("type") == "video_url")
    assert video_part["video_url"]["url"] == "https://x/clip.mp4"


def test_transform_anthropic_image_plus_audio_plus_video():
    """All three modalities in one user turn pass through together."""
    payload = {
        "model": "claude-3-5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "multi input"},
                    {"type": "image", "source": {"type": "url", "url": "https://x/a.png"}},
                    {
                        "type": "audio",
                        "source": {"type": "base64", "media_type": "audio/wav", "data": "AAA"},
                    },
                    {"type": "video", "source": {"type": "url", "url": "https://x/v.mp4"}},
                ],
            }
        ],
    }
    out = transform_anthropic_to_openai(payload)
    parts = out["messages"][0]["content"]
    types = [p.get("type") for p in parts]
    assert "text" in types
    assert "image_url" in types
    assert "input_audio" in types
    assert "video_url" in types


@pytest.fixture
def client(tmp_path):
    from potato.config import Settings

    settings = Settings(
        proxy_api_keys=["sk-test"],
        allow_insecure_auth=True,
        catalog_fetch_docs=False,
        catalog_run_probes=False,
        nim_api_keys=["nvapi-dummy"],
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_chat_endpoint_default_model(client, monkeypatch):
    async def fake_request_json(*args, **kwargs):
        return (
            200,
            {
                "id": "cmpl-1",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hello from auto router"},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            },
            {},
            None,
        )

    from potato.upstream import UpstreamClient

    monkeypatch.setattr(UpstreamClient, "request_json", fake_request_json)

    res = client.post(
        "/chat",
        headers={"Authorization": "Bearer sk-test"},
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert res.status_code == 200
    data = res.json()
    assert "choices" in data or "content" in data


def test_v1_messages_anthropic_endpoint(client, monkeypatch):
    async def fake_request_json(*args, **kwargs):
        return (
            200,
            {
                "id": "cmpl-2",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Response for Claude UI"},
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 6},
            },
            {},
            None,
        )

    from potato.upstream import UpstreamClient

    monkeypatch.setattr(UpstreamClient, "request_json", fake_request_json)

    res = client.post(
        "/v1/messages",
        headers={"x-api-key": "sk-test"},
        json={
            "model": "claude-3-5-sonnet-20241022",
            "system": "Act as a helper",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"][0]["text"] == "Response for Claude UI"
    assert data["usage"]["input_tokens"] == 12


def test_tool_call_auto_recovery():
    from potato.routes.claude import (
        _extract_raw_tool_calls_from_text,
        transform_openai_to_anthropic_json,
    )

    raw_text = "<|tool_calls_section_begin|><|tool_call:read_file{\"path\": \"src/main.py\"}|><|tool_calls_section_end|>"
    clean_text, tools = _extract_raw_tool_calls_from_text(raw_text)
    assert clean_text == ""
    assert len(tools) == 1
    assert tools[0]["type"] == "tool_use"
    assert tools[0]["name"] == "read_file"
    assert tools[0]["input"] == {"path": "src/main.py"}

    # Test full Anthropic JSON conversion with raw text tool call
    openai_resp = {
        "id": "cmpl-raw-tool",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": raw_text,
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
    }
    converted = transform_openai_to_anthropic_json(openai_resp, "potato/coding")
    assert converted["stop_reason"] == "tool_use"
    assert len(converted["content"]) == 1
    assert converted["content"][0]["type"] == "tool_use"
    assert converted["content"][0]["name"] == "read_file"

