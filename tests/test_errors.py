"""Every failure has to come back as a sentence the chat model can act on."""

import requests
from conftest import EMPTY_PAYLOAD, INCOMPLETE_PAYLOAD, arun, message_payload, thinking
from x_search import Tools


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

    def test_conflict_fires_even_when_one_side_is_unparseable(self, tool, xai):
        """Parsing first would drop the bad side and hide the mistake."""
        out = arun(tool.search_x("q", allowed_handles="nasa", excluded_handles="!!!"))
        assert "cannot be used in the same search" in out
        assert xai.calls == 0


class TestUnusableHandleFilters:
    """A filter asked for but parsed to nothing would search all of X instead.

    That is broader than what was asked for, and costs a full agent investigation
    before anyone finds out — so it is refused before the request goes out.
    """

    def test_unparseable_allowed_handles_refuse_rather_than_widen(self, tool, xai):
        out = arun(tool.search_x("q", allowed_handles="bad-handle!!"))
        assert out.startswith("Error:")
        assert "broader than asked" in out
        assert "bad-handle!!" in out
        assert xai.calls == 0

    def test_a_post_url_with_no_handle_in_it_is_refused(self, tool, xai):
        """The realistic path: a model passing back a citation from an earlier search."""
        out = arun(
            tool.search_x("q", allowed_handles="https://x.com/i/status/1975607901571199086")
        )
        assert out.startswith("Error:")
        assert xai.calls == 0

    def test_unparseable_call_level_exclusions_are_refused_too(self, tool, xai):
        out = arun(tool.search_x("q", excluded_handles="!!!"))
        assert out.startswith("Error:")
        assert "excluded_handles" in out
        assert xai.calls == 0

    def test_an_unusable_allowed_user_valve_is_refused(self, tool, xai):
        """An allow-list failing open is a scope violation whatever its source."""
        valves = Tools.UserValves(ALLOWED_HANDLES="!!!")
        out = arun(tool.search_x("q", __user__={"valves": valves}))
        assert out.startswith("Error:")
        assert xai.calls == 0

    def test_an_unusable_exclusion_default_only_warns(self, tool, xai):
        """One bad instance-wide valve must not brick every search on the server."""
        tool.valves.DEFAULT_EXCLUDED_HANDLES = "!!!"
        out = arun(tool.search_x("q"))
        assert not out.startswith("Error:")
        assert xai.calls == 1
        assert "excluded_x_handles" not in xai.tool_config
        assert "Ignored unparseable handle(s): !!!." in out


class TestTruncatedResponses:
    """A cut-short response has no message block, exactly like an empty result.

    Reporting it as "no posts found" states a falsehood about X, which the whole
    tool is otherwise careful never to do.
    """

    def test_truncation_is_not_reported_as_an_empty_result(self, tool, xai):
        xai.queue(payload=INCOMPLETE_PAYLOAD)
        out = arun(tool.search_x("what are people saying about xAI"))
        assert "No X posts were found" not in out
        assert out.startswith("Error:")
        assert "max_output_tokens" in out
        assert "not a finding about X" in out

    def test_it_points_at_the_valve_responsible(self, tool, xai):
        xai.queue(payload=INCOMPLETE_PAYLOAD)
        out = arun(tool.search_x("q"))
        assert "MAX_OUTPUT_TOKENS" in out

    def test_partial_text_is_kept_and_flagged(self, tool, xai):
        payload = message_payload(text="Partial answer.")
        payload["status"] = "incomplete"
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
        xai.queue(payload=payload)
        out = arun(tool.search_x("q"))
        assert "Partial answer." in out
        assert "cut short" in out

    def test_the_status_line_closes_out(self, tool, xai, events):
        xai.queue(payload=INCOMPLETE_PAYLOAD)
        arun(tool.search_x("q", __event_emitter__=events))
        assert events.of_type("status")[-1]["data"]["done"] is True

    def test_the_call_still_counts_against_the_budget(self, tool, xai):
        """Unlike a transport failure, this one was billed, and a retry hits the
        same cap rather than fixing anything — so the slot is not given back."""
        tool.valves.MAX_SEARCHES_PER_MESSAGE = 1
        xai.queue(payload=INCOMPLETE_PAYLOAD)

        async def scenario():
            await tool.search_x("q1", __message_id__="t1")
            await thinking()
            return await tool.search_x("q2", __message_id__="t1")

        assert arun(scenario()).startswith("Skipped:")
        assert xai.calls == 1

    def test_an_identical_repeat_is_served_from_memory(self, tool, xai):
        xai.queue(payload=INCOMPLETE_PAYLOAD)

        async def scenario():
            first = await tool.search_x("q", __message_id__="t2")
            await thinking()
            return first, await tool.search_x("q", __message_id__="t2")

        first, second = arun(scenario())
        assert first == second
        assert xai.calls == 1


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

    def test_json_that_is_not_an_object(self, tool, xai):
        """Every reader below calls .get(); a bare list would crash the tool."""
        for payload in ([], "surprise", 42):
            xai.queue(payload=payload, status=200)
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
