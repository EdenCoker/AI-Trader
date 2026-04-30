import json

import httpx
import pytest
from pydantic import SecretStr

from ai_trader.config import AppSettings
from ai_trader.llm.errors import LLMConfigurationError
from ai_trader.llm.factory import get_llm_client
from ai_trader.llm.ollama import OllamaClient
from ai_trader.llm.openai_compatible import OpenAICompatibleClient


def test_openai_compatible_complete_calls_chat_completions():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://llm.test/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"

        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "gpt-test"
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "sys"
        assert body["messages"][1]["role"] == "user"
        assert body["messages"][1]["content"] == "hi"

        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    llm = OpenAICompatibleClient(
        base_url="https://llm.test/v1",
        api_key="test-key",
        default_model="gpt-test",
        http_client=http_client,
    )

    assert llm.complete("hi", system="sys") == "ok"


def test_ollama_complete_calls_api_chat():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "http://ollama.test/api/chat"

        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "llama-test"
        assert body["stream"] is False
        assert body["options"]["temperature"] == pytest.approx(0.33)
        assert body["options"]["num_predict"] == 123
        assert body["messages"][0]["role"] == "user"

        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "local-ok"}, "done": True},
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    llm = OllamaClient(
        base_url="http://ollama.test",
        default_model="llama-test",
        http_client=http_client,
    )

    assert llm.complete("hi", temperature=0.33, max_tokens=123) == "local-ok"


def test_factory_requires_key_for_openai_backend():
    settings = AppSettings(
        llm_backend="openai",
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        llm_model="gpt-4.5",
    )
    with pytest.raises(LLMConfigurationError):
        get_llm_client(settings)


def test_factory_creates_ollama_client():
    settings = AppSettings(
        llm_backend="ollama",
        ollama_base_url="http://localhost:11434",
        llm_model="llama3.1",
        llm_timeout_s=1.0,
    )
    llm = get_llm_client(settings)
    assert isinstance(llm, OllamaClient)


def test_factory_creates_openai_compatible_client_without_key():
    settings = AppSettings(
        llm_backend="openai_compatible",
        openai_api_key=None,
        openai_base_url="http://localhost:8000/v1",
        llm_model="some-local-model",
        llm_timeout_s=1.0,
    )
    llm = get_llm_client(settings)
    assert isinstance(llm, OpenAICompatibleClient)


def test_factory_creates_openai_client_with_key():
    settings = AppSettings(
        llm_backend="openai",
        openai_api_key=SecretStr("sk-test"),
        openai_base_url="https://api.openai.com/v1",
        llm_model="gpt-4.5",
        llm_timeout_s=1.0,
    )
    llm = get_llm_client(settings)
    assert isinstance(llm, OpenAICompatibleClient)

