"""The request the tool builds, and the result it hands back to the chat model."""

import inspect
from datetime import datetime, timedelta, timezone

from conftest import arun, message_payload
from x_search import Tools


TODAY = datetime.now(timezone.utc).date()


class TestToolSurface:
    def test_exactly_one_tool_is_exposed(self):
        """Two overlapping tools invited models to call both for one question."""
        exposed = sorted(
            name
            for name in dir(Tools)
            if callable(getattr(Tools, name))
            and not name.startswith("_")
            and not inspect.isclass(getattr(Tools, name))
        )
        assert exposed == ["search_x"]

    def test_safe_under_the_older_discovery_rule(self):
        """Older Open WebUI only excluded dunder names, so helpers live at module level."""
        exposed = sorted(
            name
            for name in dir(Tools)
            if callable(getattr(Tools, name))
            and not name.startswith("__")
            and not isinstance(getattr(Tools, name), type)
        )
        assert exposed == ["search_x"]

    def test_every_parameter_is_documented(self):
        """Open WebUI builds the model-visible spec from ':param name:' lines."""
        doc = inspect.getdoc(Tools.search_x)
        parameters = [
            name
            for name in inspect.signature(Tools.search_x).parameters
            if name != "self" and not name.startswith("__")
        ]
        assert parameters
        for name in parameters:
            assert f":param {name}:" in doc


class TestRequest:
    def test_endpoint_auth_and_body(self, tool, xai):
        arun(tool.search_x("what are people saying about xAI"))

        request = xai.requests[-1]
        assert request["url"] == "https://api.x.ai/v1/responses"
        assert request["headers"]["Authorization"] == "Bearer test-key"
        assert request["timeout"] == tool.valves.TIMEOUT_SECONDS

        body = request["body"]
        assert body["model"] == "grok-4.5"
        assert body["input"] == [
            {"role": "user", "content": "what are people saying about xAI"}
        ]
        assert body["tools"] == [{"type": "x_search"}]
        assert "instructions" in body

    def test_base_url_trailing_slash_is_handled(self, tool, xai):
        tool.valves.BASE_URL = "https://proxy.local/v1/"
        arun(tool.search_x("q"))
        assert xai.requests[-1]["url"] == "https://proxy.local/v1/responses"

    def test_optional_body_fields_are_omitted_by_default(self, tool, xai):
        arun(tool.search_x("q"))
        assert "max_output_tokens" not in xai.body

    def test_max_output_tokens_valve(self, tool, xai):
        tool.valves.MAX_OUTPUT_TOKENS = 1500
        arun(tool.search_x("q"))
        assert xai.body["max_output_tokens"] == 1500

    def test_empty_instructions_are_omitted(self, tool, xai):
        tool.valves.INSTRUCTIONS = ""
        arun(tool.search_x("q"))
        assert "instructions" not in xai.body


class TestToolConfig:
    def test_allowed_handles(self, tool, xai):
        arun(tool.search_x("q", allowed_handles="@xai, elonmusk"))
        assert xai.tool_config["allowed_x_handles"] == ["xai", "elonmusk"]
        assert "excluded_x_handles" not in xai.tool_config

    def test_excluded_handles(self, tool, xai):
        arun(tool.search_x("q", excluded_handles="spammer"))
        assert xai.tool_config["excluded_x_handles"] == ["spammer"]

    def test_explicit_date_range(self, tool, xai):
        arun(tool.search_x("q", from_date="2026-01-01", to_date="2026-02-01"))
        assert xai.tool_config["from_date"] == "2026-01-01"
        assert xai.tool_config["to_date"] == "2026-02-01"

    def test_media_understanding_is_off_by_default(self, tool, xai):
        arun(tool.search_x("q"))
        assert "enable_image_understanding" not in xai.tool_config
        assert "enable_video_understanding" not in xai.tool_config

    def test_media_understanding_valves(self, tool, xai):
        tool.valves.ENABLE_IMAGE_UNDERSTANDING = True
        tool.valves.ENABLE_VIDEO_UNDERSTANDING = True
        arun(tool.search_x("q"))
        assert xai.tool_config["enable_image_understanding"] is True
        assert xai.tool_config["enable_video_understanding"] is True

    def test_extra_tool_params_are_merged(self, tool, xai):
        tool.valves.EXTRA_TOOL_PARAMS = '{"max_search_results": 30}'
        arun(tool.search_x("q"))
        assert xai.tool_config["max_search_results"] == 30


class TestRecency:
    def test_days_sets_the_window(self, tool, xai):
        out = arun(tool.search_x("breaking news", days=3))
        assert xai.tool_config["from_date"] == (TODAY - timedelta(days=2)).isoformat()
        assert xai.tool_config["to_date"] == TODAY.isoformat()
        assert "last 3 days" in out

    def test_days_of_one_is_today(self, tool, xai):
        out = arun(tool.search_x("q", days=1))
        assert xai.tool_config["from_date"] == TODAY.isoformat()
        assert "today" in out

    def test_days_zero_means_no_date_filter(self, tool, xai):
        arun(tool.search_x("q", days=0))
        assert "from_date" not in xai.tool_config
        assert "to_date" not in xai.tool_config

    def test_string_days_are_coerced(self, tool, xai):
        arun(tool.search_x("q", days="5"))
        assert xai.tool_config["from_date"] == (TODAY - timedelta(days=4)).isoformat()

    def test_nonsense_days_fall_back_to_no_filter(self, tool, xai):
        arun(tool.search_x("q", days="soon"))
        assert "from_date" not in xai.tool_config

    def test_days_is_capped_at_a_year(self, tool, xai):
        arun(tool.search_x("q", days=5000))
        assert xai.tool_config["from_date"] == (TODAY - timedelta(days=364)).isoformat()

    def test_explicit_dates_override_days(self, tool, xai):
        arun(tool.search_x("q", days=30, from_date="2026-01-01", to_date="2026-01-05"))
        assert xai.tool_config["from_date"] == "2026-01-01"
        assert xai.tool_config["to_date"] == "2026-01-05"


class TestUserValves:
    def test_user_excluded_handles_apply(self, tool, xai):
        valves = Tools.UserValves(EXCLUDED_HANDLES="spammer1, @spammer2")
        arun(tool.search_x("q", __user__={"valves": valves}))
        assert xai.tool_config["excluded_x_handles"] == ["spammer1", "spammer2"]

    def test_user_media_valve_applies(self, tool, xai):
        valves = Tools.UserValves(ENABLE_MEDIA_UNDERSTANDING=True)
        arun(tool.search_x("q", __user__={"valves": valves}))
        assert xai.tool_config["enable_image_understanding"] is True
        assert xai.tool_config["enable_video_understanding"] is True

    def test_call_level_allowed_handles_win_over_user_exclusions(self, tool, xai):
        """The API rejects both filters, so the more specific one has to win."""
        valves = Tools.UserValves(EXCLUDED_HANDLES="spammer")
        out = arun(tool.search_x("q", allowed_handles="nasa", __user__={"valves": valves}))
        assert xai.tool_config["allowed_x_handles"] == ["nasa"]
        assert "excluded_x_handles" not in xai.tool_config
        assert "restricted to specific handles" in out

    def test_default_excluded_handles_valve(self, tool, xai):
        tool.valves.DEFAULT_EXCLUDED_HANDLES = "noise"
        arun(tool.search_x("q"))
        assert xai.tool_config["excluded_x_handles"] == ["noise"]


class TestResult:
    def test_includes_summary_scope_and_queries(self, tool, xai):
        xai.queue(
            payload=message_payload(
                text="People are positive.",
                searches=["xai reaction", "grok launch"],
            )
        )
        out = arun(tool.search_x("reaction to Grok", from_date="2026-07-01"))

        assert "People are positive." in out
        assert "Scope: 2026-07-01" in out
        assert "Queries run on X: xai reaction; grok launch" in out

    def test_cited_sources_reuse_the_models_numbering(self, tool, xai):
        """A '[2]' in the summary must point at entry [2] in the list."""
        xai.queue(
            payload=message_payload(
                text="A [[1]](...) and B [[2]](...).",
                cited=(
                    ("https://x.com/a/status/1", "1"),
                    ("https://x.com/i/status/2", "2"),
                ),
                extra_citations=("https://x.ai/news",),
            )
        )
        out = arun(tool.search_x("q"))

        assert "Sources cited above:" in out
        assert "[1] https://x.com/a/status/1 (@a)" in out
        assert "[2] https://x.com/i/status/2 (X post)" in out
        assert "Other sources consulted (not cited in the summary):" in out
        assert "- https://x.ai/news" in out

    def test_emits_citation_events(self, tool, xai, events):
        xai.queue(
            payload=message_payload(
                cited=(("https://x.com/elonmusk/status/1", "1"),),
                extra_citations=("https://x.ai/news",),
            )
        )
        arun(tool.search_x("q", __event_emitter__=events))

        citations = events.of_type("citation")
        assert len(citations) == 2
        names = [event["data"]["source"]["name"] for event in citations]
        assert names == ["@elonmusk", "x.ai (consulted)"]

    def test_emits_start_and_finish_status(self, tool, xai, events):
        arun(tool.search_x("q", __event_emitter__=events))
        statuses = events.of_type("status")
        assert len(statuses) >= 2
        assert statuses[0]["data"]["done"] is False
        assert statuses[-1]["data"]["done"] is True

    def test_events_can_be_switched_off(self, tool, xai, events):
        tool.valves.EMIT_STATUS = False
        tool.valves.EMIT_CITATIONS = False
        arun(tool.search_x("q", __event_emitter__=events))
        assert events.events == []
