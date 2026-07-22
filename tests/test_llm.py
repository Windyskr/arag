from arag.core.llm import LLMClient


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def test_llm_client_retries_rate_limits(monkeypatch):
    responses = iter([
        FakeResponse(429, headers={"Retry-After": "0"}),
        FakeResponse(200, {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }),
    ])
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return next(responses)

    monkeypatch.setattr("arag.core.llm.requests.post", fake_post)
    monkeypatch.setattr("arag.core.llm.time.sleep", lambda delay: None)
    monkeypatch.setattr("arag.core.llm.random.random", lambda: 0.0)

    client = LLMClient(api_key="test", max_retries=2)
    result = client.chat([{"role": "user", "content": "hello"}])
    assert result["message"]["content"] == "ok"
    assert len(calls) == 2


def test_llm_client_does_not_retry_model_not_found(monkeypatch):
    import pytest
    import requests

    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse(404, text='{"error":{"type":"model_not_found"}}')

    monkeypatch.setattr("arag.core.llm.requests.post", fake_post)
    monkeypatch.setattr("arag.core.llm.time.sleep", lambda delay: None)

    client = LLMClient(api_key="test", max_retries=3)
    with pytest.raises(requests.HTTPError, match="HTTP 404") as exc_info:
        client.chat([{"role": "user", "content": "hello"}])

    assert exc_info.value.response.status_code == 404
    assert len(calls) == 1


def test_llm_client_does_not_retry_permanent_reasoning_content_error(monkeypatch):
    import pytest
    import requests

    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse(
            400,
            text='{"error":{"message":"The reasoning_content in the thinking mode must be passed back to the API.","type":"invalid_request_error"}}',
        )

    monkeypatch.setattr("arag.core.llm.requests.post", fake_post)
    monkeypatch.setattr("arag.core.llm.time.sleep", lambda delay: None)

    client = LLMClient(api_key="test", max_retries=3)
    with pytest.raises(requests.HTTPError, match="reasoning_content") as exc_info:
        client.chat([{"role": "user", "content": "hello"}])

    assert exc_info.value.response.status_code == 400
    assert len(calls) == 1
