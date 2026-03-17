"""Tests for LiteLLM Proxy provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.config.schema import Config
from nanobot.providers.litellm_proxy_provider import LiteLLMProxyProvider


def test_provider_stores_defaults():
    provider = LiteLLMProxyProvider(
        api_key="sk-test",
        api_base="http://localhost:4000",
        default_model="my-model",
    )
    assert provider.get_default_model() == "my-model"
    assert provider.api_key == "sk-test"
    assert provider.api_base == "http://localhost:4000"


def test_provider_accepts_extra_headers():
    provider = LiteLLMProxyProvider(
        api_key="sk-test",
        api_base="http://localhost:4000",
        default_model="my-model",
        extra_headers={"x-custom": "value"},
    )
    # AsyncOpenAI stores custom headers; verify they're accessible
    assert provider.extra_headers == {"x-custom": "value"}


@pytest.mark.asyncio
async def test_chat_sends_correct_kwargs():
    provider = LiteLLMProxyProvider(
        api_key="sk-test",
        api_base="http://localhost:4000",
        default_model="my-model",
    )

    mock_message = MagicMock()
    mock_message.content = "Hello!"
    mock_message.tool_calls = None
    mock_message.reasoning_content = None

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await provider.chat(
        messages=[{"role": "user", "content": "Hi"}],
        model="test-model",
        max_tokens=100,
        temperature=0.5,
    )

    call_kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["max_tokens"] == 100
    assert call_kwargs["temperature"] == 0.5
    assert result.content == "Hello!"
    assert result.finish_reason == "stop"
    assert result.usage["total_tokens"] == 15


@pytest.mark.asyncio
async def test_chat_uses_default_model_when_none():
    provider = LiteLLMProxyProvider(
        api_key="sk-test",
        api_base="http://localhost:4000",
        default_model="fallback-model",
    )

    mock_message = MagicMock()
    mock_message.content = "ok"
    mock_message.tool_calls = None
    mock_message.reasoning_content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

    await provider.chat(messages=[{"role": "user", "content": "Hi"}])

    call_kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "fallback-model"


@pytest.mark.asyncio
async def test_chat_includes_tools_when_provided():
    provider = LiteLLMProxyProvider(
        api_key="sk-test",
        api_base="http://localhost:4000",
        default_model="my-model",
    )

    mock_message = MagicMock()
    mock_message.content = "ok"
    mock_message.tool_calls = None
    mock_message.reasoning_content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

    tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
    await provider.chat(messages=[{"role": "user", "content": "Hi"}], tools=tools)

    call_kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_chat_returns_error_on_exception():
    provider = LiteLLMProxyProvider(
        api_key="sk-test",
        api_base="http://localhost:4000",
        default_model="my-model",
    )

    provider._client.chat.completions.create = AsyncMock(
        side_effect=ConnectionError("proxy unreachable")
    )

    result = await provider.chat(messages=[{"role": "user", "content": "Hi"}])

    assert result.finish_reason == "error"
    assert "proxy unreachable" in result.content


@pytest.mark.asyncio
async def test_chat_parses_tool_calls():
    provider = LiteLLMProxyProvider(
        api_key="sk-test",
        api_base="http://localhost:4000",
        default_model="my-model",
    )

    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "read_file"
    mock_tc.function.arguments = '{"path": "/tmp/test.txt"}'

    mock_message = MagicMock()
    mock_message.content = None
    mock_message.tool_calls = [mock_tc]
    mock_message.reasoning_content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = "tool_calls"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    provider._client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await provider.chat(messages=[{"role": "user", "content": "read file"}])

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "/tmp/test.txt"}
    assert result.finish_reason == "tool_calls"


def test_config_matches_litellm_proxy_with_explicit_provider():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "litellm_proxy", "model": "my-proxy-model"}},
            "providers": {"litellm_proxy": {"apiBase": "http://localhost:4000", "apiKey": "sk-test"}},
        }
    )

    assert config.get_provider_name() == "litellm_proxy"
    p = config.get_provider()
    assert p.api_base == "http://localhost:4000"
    assert p.api_key == "sk-test"


def test_config_matches_litellm_proxy_with_auto_and_keyword_prefix():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "auto", "model": "litellm-proxy/claude-sonnet"}},
            "providers": {"litellm_proxy": {"apiBase": "http://localhost:4000"}},
        }
    )

    assert config.get_provider_name() == "litellm_proxy"
