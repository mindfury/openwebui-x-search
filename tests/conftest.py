"""Shared fixtures and a stand-in for the xAI API.

The tool is a single module pasted into Open WebUI, so the tests import it directly
and replace `requests.post` rather than starting a server.
"""

import asyncio

import pytest

import x_search
from x_search import Tools


def message_payload(
    text="Summary text.",
    cited=(("https://x.com/a/status/1", "1"),),
    extra_citations=(),
    searches=(),
):
    """A Responses API payload shaped like the one xAI documents."""
    output = [{"type": "x_search_call", "action": {"query": query}} for query in searches]
    output.append(
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": text,
                    "annotations": [
                        {"type": "url_citation", "url": url, "title": title}
                        for url, title in cited
                    ],
                }
            ],
        }
    )
    payload = {"output": output}
    if extra_citations:
        payload["citations"] = list(extra_citations)
    return payload


EMPTY_PAYLOAD = {"output": [], "citations": []}


class FakeResponse:
    def __init__(self, status_code, payload, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or "response body"

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeXAI:
    """Records outgoing requests and returns queued responses."""

    def __init__(self):
        self.requests = []
        self._queue = []

    def queue(self, payload=None, status=200, raises=None, text=""):
        self._queue.append((status, payload, raises, text))
        return self

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append(
            {"url": url, "headers": headers, "body": json, "timeout": timeout}
        )
        if self._queue:
            status, payload, raises, text = self._queue.pop(0)
        else:
            status, payload, raises, text = 200, message_payload(), None, ""
        if raises is not None:
            raise raises
        return FakeResponse(status, payload, text)

    @property
    def calls(self):
        return len(self.requests)

    @property
    def body(self):
        return self.requests[-1]["body"]

    @property
    def tool_config(self):
        return self.body["tools"][0]

    @property
    def queries(self):
        return [request["body"]["input"][0]["content"] for request in self.requests]


class Recorder:
    """Captures the events the tool emits to Open WebUI."""

    def __init__(self):
        self.events = []

    async def __call__(self, event):
        self.events.append(event)

    def of_type(self, event_type):
        return [event for event in self.events if event["type"] == event_type]


@pytest.fixture(autouse=True)
def clean_gate_state():
    """The gate keeps module-level state; no test may leak into another."""
    x_search._MESSAGE_CALLS.clear()
    yield
    x_search._MESSAGE_CALLS.clear()


@pytest.fixture
def xai(monkeypatch):
    fake = FakeXAI()
    monkeypatch.setattr(x_search.requests, "post", fake.post)
    return fake


@pytest.fixture
def tool():
    instance = Tools()
    instance.valves.XAI_API_KEY = "test-key"
    # Tests exercise gating explicitly; keep the batch window small so a simulated
    # "model thought about it" pause stays fast.
    instance.valves.BATCH_WINDOW_SECONDS = 0.2
    return instance


@pytest.fixture
def events():
    return Recorder()


def arun(coro):
    """Run one coroutine to completion."""
    return asyncio.run(coro)


async def thinking():
    """Stand in for the model inference that precedes a genuine retry."""
    await asyncio.sleep(0.35)
