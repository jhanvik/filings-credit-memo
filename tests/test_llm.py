"""Tests for the LLM wrapper classes. No real model or network is used."""

import json

import httpx
import pytest

from filings_credit_memo.llm import CachedLLM, LLMError, OllamaLLM


class CountingLLM:
    name = "fake"

    def __init__(self, reply="OK"):
        self.reply = reply
        self.seen = []

    def complete(self, system, user, schema=None):
        self.seen.append((system, user, schema))
        return self.reply


def ollama(handler):
    return OllamaLLM("test-model", "http://ollama.test", transport=httpx.MockTransport(handler))


def test_ollama_sends_expected_request_and_returns_message_content():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": '{"items": []}'}})

    schema = {"type": "object"}
    assert ollama(handler).complete("sys", "user text", schema) == '{"items": []}'
    body = captured["body"]
    assert captured["url"] == "http://ollama.test/api/chat"
    assert body["model"] == "test-model" and body["stream"] is False and body["format"] == schema
    assert body["options"]["temperature"] == 0 and body["options"]["seed"] == 42
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_ollama_omits_format_when_no_schema():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "hi"}})

    ollama(handler).complete("s", "u")
    assert "format" not in seen


def test_ollama_http_error_and_bad_body_become_llm_errors():
    with pytest.raises(LLMError, match="HTTP 500"):
        ollama(lambda r: httpx.Response(500, text="boom")).complete("s", "u")
    with pytest.raises(LLMError, match="unexpected response"):
        ollama(lambda r: httpx.Response(200, json={"nope": 1})).complete("s", "u")


def test_ollama_connection_failure_gives_actionable_hint():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="brew services start ollama"):
        ollama(handler).complete("s", "u")


def test_cached_llm_answers_repeat_requests_from_disk_and_logs_the_prompt(tmp_path):
    inner = CountingLLM("answer")
    llm = CachedLLM(inner, tmp_path / "cache")
    assert llm.complete("s", "u", {"a": 1}) == "answer"
    assert llm.complete("s", "u", {"a": 1}) == "answer"
    assert len(inner.seen) == 1 and llm.calls == 1
    record = json.loads(next((tmp_path / "cache").glob("*.json")).read_text())
    assert record["user"] == "u" and record["response"] == "answer" and record["model"] == "fake"


def test_cache_key_depends_on_every_input(tmp_path):
    inner = CountingLLM()
    llm = CachedLLM(inner, tmp_path / "c")
    llm.complete("s", "u", None)
    llm.complete("s2", "u", None)
    llm.complete("s", "u2", None)
    llm.complete("s", "u", {"x": 1})
    assert len(inner.seen) == 4


def test_cache_survives_a_new_wrapper_instance(tmp_path):
    CachedLLM(CountingLLM("first"), tmp_path / "c").complete("s", "u")
    second = CountingLLM("second")
    assert CachedLLM(second, tmp_path / "c").complete("s", "u") == "first"
    assert second.seen == []
