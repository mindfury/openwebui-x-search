"""Every failure has to come back as a sentence the chat model can act on."""

import requests
from conftest import EMPTY_PAYLOAD, arun


class TestConfiguration:
    def test_missing_api_key(self, tool, xai):
        tool.valves.XAI_API_KEY = ""
        out = arun(tool.search_x("q"))
        assert out.startswith("Error: no xAI API key")
        assert xai.calls == 0

    def test_invalid_extra_tool_params(self, tool, xai):
        tool.valves.EXTRA_TOOL_PARAMS = "{not json}"
        out = arun(tool.search_x("q"))
        assert "not a valid JSON object" in out
        assert xai.calls == 0

    def test_extra_tool_params_must_be_an_object(self, tool, xai):
        tool.valves.EXTRA_TOOL_PARAMS = "[1, 2]"
        out = arun(tool.search_x("q"))
        assert "not a valid JSON object" in out
        assert xai.calls == 0


class TestArguments:
    def test_empty_query(self, tool, xai):
        assert arun(tool.search_x("   ")) == "Error: `query` is required."
        assert xai.calls == 0

    def test_handle_filters_are_mutually_exclusive(self, tool, xai):
        out = arun(tool.search_x("q", allowed_handles="a", excluded_handles="b"))
        assert "cannot be used in the same search" in out
        assert xai.calls == 0

    def test_malformed_date(self, tool, xai):
        out = arun(tool.search_x("q", from_date="08/01/2026"))
        assert "YYYY-MM-DD" in out
        assert xai.calls == 0

    def test_inverted_date_range(self, tool, xai):
        out = arun(tool.search_x("q", from_date="2026-08-05", to_date="2026-08-01"))
        assert "is after" in out
        assert xai.calls == 0


class TestUpstreamFailures:
    def test_bad_key(self, tool, xai):
        xai.queue(payload={"error": {"message": "invalid api key"}}, status=401)
        out = arun(tool.search_x("q"))
        assert "rejected the API key" in out
        assert "invalid api key" in out

    def test_rate_limited(self, tool, xai):
        xai.queue(payload={"error": "slow down"}, status=429)
        out = arun(tool.search_x("q"))
        assert "rate limit" in out

    def test_unknown_model(self, tool, xai):
        xai.queue(payload={"error": {"message": "no such model"}}, status=404)
        out = arun(tool.search_x("q"))
        assert "not found" in out
        assert "MODEL valve" in out

    def test_server_error(self, tool, xai):
        xai.queue(payload={"error": {"message": "boom"}}, status=500)
        out = arun(tool.search_x("q"))
        assert "returned 500" in out

    def test_non_json_response(self, tool, xai):
        xai.queue(payload=None, status=200)
        out = arun(tool.search_x("q"))
        assert "not valid JSON" in out

    def test_timeout(self, tool, xai):
        xai.queue(raises=requests.Timeout("slow"))
        out = arun(tool.search_x("q"))
        assert "timed out" in out
        assert "TIMEOUT_SECONDS" in out

    def test_connection_error(self, tool, xai):
        xai.queue(raises=requests.ConnectionError("no route"))
        out = arun(tool.search_x("q"))
        assert "could not reach the xAI API" in out

    def test_failures_still_close_out_the_status(self, tool, xai, events):
        xai.queue(raises=requests.ConnectionError("no route"))
        arun(tool.search_x("q", __event_emitter__=events))
        assert events.of_type("status")[-1]["data"]["done"] is True


class TestEmptyResults:
    def test_reports_nothing_found_rather_than_inventing(self, tool, xai):
        xai.queue(payload=EMPTY_PAYLOAD)
        out = arun(tool.search_x("something obscure"))
        assert "No X posts were found" in out

    def test_suggests_widening_when_filters_were_applied(self, tool, xai):
        xai.queue(payload=EMPTY_PAYLOAD)
        out = arun(tool.search_x("q", from_date="2026-01-01"))
        assert "widening" in out

    def test_no_such_hint_without_filters(self, tool, xai):
        xai.queue(payload=EMPTY_PAYLOAD)
        out = arun(tool.search_x("q"))
        assert "widening" not in out
