"""Pure helpers: handle parsing, dates, and reading xAI's response shape."""

import pytest

import x_search


class TestHandles:
    def test_accepts_bare_at_and_url_forms(self):
        handles, _ = x_search._normalise_handles(
            "@elonmusk, https://x.com/xai , twitter.com/grok, x.com/foo/status/1"
        )
        assert handles == ["elonmusk", "xai", "grok", "foo"]

    def test_whitespace_is_a_separator(self):
        handles, _ = x_search._normalise_handles("@a @b")
        assert handles == ["a", "b"]

    def test_warns_about_unparseable_handles(self):
        handles, warning = x_search._normalise_handles(
            "elonmusk, bad-handle!!, a_handle_far_too_long_to_be_real"
        )
        assert handles == ["elonmusk"]
        assert "bad-handle!!" in warning
        assert "a_handle_far_too_long_to_be_real" in warning

    def test_caps_at_the_api_limit_of_20(self):
        handles, warning = x_search._normalise_handles(
            ",".join(f"user{index}" for index in range(30))
        )
        assert len(handles) == x_search.MAX_HANDLES == 20
        assert "20" in warning

    def test_dedupes_case_insensitively_keeping_first_spelling(self):
        handles, _ = x_search._normalise_handles("@xAI, xai, XAI")
        assert handles == ["xAI"]

    def test_empty_input(self):
        assert x_search._normalise_handles("") == ([], None)

    def test_rejects_urls_built_on_reserved_paths(self):
        """xAI's own citations use the handle-less x.com/i/status/<id> form."""
        handles, warning = x_search._normalise_handles(
            "https://x.com/i/status/1975607901571199086, x.com/i/user/1912644073896206336"
        )
        assert handles == []
        assert "carry no handle" in warning

    def test_reserved_paths_are_dropped_alongside_real_handles(self):
        handles, warning = x_search._normalise_handles("@nasa, x.com/home, x.com/xai")
        assert handles == ["nasa", "xai"]
        assert "x.com/home" in warning

    def test_a_bare_reserved_word_is_still_a_handle(self):
        """Only a URL path segment can be reserved; @home is an ordinary account."""
        assert x_search._normalise_handles("@home")[0] == ["home"]

    def test_rejects_rather_than_truncates_an_over_long_url_handle(self):
        """Truncating would yield a wrong but entirely plausible handle."""
        handles, warning = x_search._normalise_handles(
            "x.com/a_handle_far_too_long_to_be_real"
        )
        assert handles == []
        assert warning is not None

    def test_url_handles_may_be_followed_by_a_delimiter(self):
        handles, warning = x_search._normalise_handles(
            "https://x.com/elonmusk?lang=en https://x.com/xai#bio x.com/foo/status/1"
        )
        assert handles == ["elonmusk", "xai", "foo"]
        assert warning is None

    def test_host_matching_is_case_insensitive(self):
        assert x_search._normalise_handles("HTTPS://X.COM/elonmusk")[0] == ["elonmusk"]


class TestDates:
    def test_accepts_iso8601(self):
        assert x_search._validate_date("2026-08-01", "from_date") == ("2026-08-01", None)

    def test_rejects_other_formats(self):
        value, error = x_search._validate_date("08/01/2026", "from_date")
        assert value == ""
        assert "YYYY-MM-DD" in error

    def test_rejects_impossible_dates(self):
        _, error = x_search._validate_date("2026-13-45", "from_date")
        assert error is not None

    def test_empty_is_not_an_error(self):
        assert x_search._validate_date("", "to_date") == ("", None)


class TestSourceName:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://x.com/elonmusk/status/9", "@elonmusk"),
            ("https://x.com/elonmusk", "@elonmusk"),
            ("https://twitter.com/xai/status/9", "@xai"),
            # '/i/' is a reserved path, not a handle. These forms appear in xAI's
            # own documented citation examples.
            ("https://x.com/i/status/1975607901571199086", "X post"),
            ("https://x.com/i/user/1912644073896206336", "X profile"),
            ("https://x.ai/news", "x.ai"),
            ("https://docs.x.ai/developers/release-notes", "docs.x.ai"),
            # Other reserved paths are no more a handle than '/i/' is.
            ("https://x.com/search?q=grok", "X profile"),
            ("https://x.com/home", "X profile"),
            # xAI is not obliged to hand back a lowercased host.
            ("https://X.com/elonmusk", "@elonmusk"),
            ("HTTPS://TWITTER.COM/xai/status/9", "@xai"),
        ],
    )
    def test_labels(self, url, expected):
        assert x_search._source_name(url) == expected


class TestExtractText:
    def test_reads_message_blocks(self):
        payload = {
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "The answer."}],
                },
            ]
        }
        assert x_search._extract_text(payload) == "The answer."

    def test_falls_back_to_output_text(self):
        assert x_search._extract_text({"output_text": "Direct.", "output": []}) == "Direct."

    def test_tolerates_malformed_items(self):
        payload = {"output": ["not a dict", {"type": "message", "content": "not a list"}]}
        assert x_search._extract_text(payload) == ""


class TestExtractCitations:
    def test_keeps_the_models_own_numbering(self):
        """Inline annotations carry the number the summary text refers to."""
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "A [[1]](...) and B [[2]](...).",
                            "annotations": [
                                {"type": "url_citation", "url": "https://x.com/a/status/1", "title": "1"},
                                {"type": "url_citation", "url": "https://x.com/b/status/2", "title": "2"},
                            ],
                        }
                    ],
                }
            ],
            # Top-level citations is every source *encountered*, in arbitrary order.
            "citations": [
                "https://x.ai/news",
                "https://x.com/b/status/2",
                "https://x.com/i/user/999",
                "https://x.com/a/status/1",
            ],
        }
        citations = x_search._extract_citations(payload)

        cited = [(c["url"], c["label"]) for c in citations if c["cited"]]
        assert cited == [
            ("https://x.com/a/status/1", "1"),
            ("https://x.com/b/status/2", "2"),
        ]

        consulted = [c["url"] for c in citations if not c["cited"]]
        assert consulted == ["https://x.ai/news", "https://x.com/i/user/999"]

    def test_sorts_out_of_order_annotation_numbers(self):
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "x",
                            "annotations": [
                                {"type": "url_citation", "url": "https://x.com/c/status/3", "title": "2"},
                                {"type": "url_citation", "url": "https://x.com/d/status/4", "title": "1"},
                            ],
                        }
                    ],
                }
            ]
        }
        assert [c["label"] for c in x_search._extract_citations(payload)] == ["1", "2"]

    def test_handles_dict_shaped_citations(self):
        payload = {"citations": [{"url": "https://x.com/e/status/5", "title": "T"}]}
        citations = x_search._extract_citations(payload)
        assert citations == [{"url": "https://x.com/e/status/5", "label": "", "cited": False}]

    def test_ignores_non_http_values(self):
        assert x_search._extract_citations({"citations": ["not-a-url", None, 42]}) == []

    def test_page_titles_are_not_mistaken_for_numbers(self):
        """`title` is a page title in the general Responses API shape.

        Reusing one as a citation number would print it where a number belongs and
        break the summary's [1]/[2] cross-references entirely.
        """
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "A [1] and B [2].",
                            "annotations": [
                                {"url": "https://x.com/a/status/1", "title": "Elon Musk on X"},
                                {"url": "https://x.com/b/status/2", "title": "xAI on X"},
                            ],
                        }
                    ],
                }
            ]
        }
        assert [c["label"] for c in x_search._extract_citations(payload)] == ["", ""]

    def test_numbering_is_all_or_nothing(self):
        """A partial set would collide with the positional fallback."""
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "x",
                            "annotations": [
                                {"url": "https://x.com/a/status/1", "title": "2"},
                                {"url": "https://x.com/b/status/2"},
                            ],
                        }
                    ],
                }
            ]
        }
        assert [c["label"] for c in x_search._extract_citations(payload)] == ["", ""]


class TestIncompleteReason:
    def test_none_for_a_complete_response(self):
        assert x_search._incomplete_reason({"status": "completed", "output": []}) is None
        assert x_search._incomplete_reason({"output": []}) is None

    def test_reads_the_documented_reason(self):
        payload = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        }
        assert x_search._incomplete_reason(payload) == "max_output_tokens"

    def test_falls_back_when_no_reason_is_given(self):
        assert x_search._incomplete_reason({"status": "incomplete"}) == "reason unspecified"
        assert (
            x_search._incomplete_reason({"status": "incomplete", "incomplete_details": None})
            == "reason unspecified"
        )


class TestExtractSearches:
    def test_reads_the_queries_the_agent_ran(self):
        payload = {
            "output": [
                {"type": "x_search_call", "action": {"query": "first"}},
                {"type": "x_search_call", "query": "second"},
                {"type": "x_search_call", "arguments": '{"query": "third"}'},
                {"type": "message", "content": []},
            ]
        }
        assert x_search._extract_searches(payload) == ["first", "second", "third"]

    def test_dedupes_and_ignores_malformed_arguments(self):
        payload = {
            "output": [
                {"type": "x_search_call", "action": {"query": "same"}},
                {"type": "x_search_call", "action": {"query": "same"}},
                {"type": "x_search_call", "arguments": "{not json"},
            ]
        }
        assert x_search._extract_searches(payload) == ["same"]


class TestResolveMessageId:
    def test_prefers_the_injected_message_id(self):
        assert x_search._resolve_message_id("abc", {"message_id": "xyz"}) == "abc"

    def test_falls_back_to_metadata(self):
        assert x_search._resolve_message_id(None, {"message_id": "xyz"}) == "xyz"
        assert x_search._resolve_message_id(None, {"assistant_message_id": "aid"}) == "aid"

    def test_none_when_unavailable(self):
        assert x_search._resolve_message_id(None, None) is None
        assert x_search._resolve_message_id("", {}) is None
