"""The per-message call gate.

Models fan one question out into several tool calls issued together, before any
results exist to justify them. Open WebUI runs those serially, so the batch is
recognisable by timing: a sibling arrives the instant the previous call returns,
whereas a real retry needs a full model inference first. The gate refuses the
former and allows the latter.
"""

import asyncio
import threading

import requests
from conftest import EMPTY_PAYLOAD, arun, message_payload, thinking

import x_search


def is_skip(text):
    return text.startswith("Skipped:")


class TestFanOutBatch:
    def test_back_to_back_calls_cost_one_request(self, tool, xai):
        """Exactly how Open WebUI runs a model's parallel tool calls."""

        async def batch():
            first = await tool.search_x("reaction to the launch", __message_id__="m1")
            second = await tool.search_x("what people think of it", __message_id__="m1")
            return first, second

        first, second = arun(batch())

        assert xai.calls == 1
        assert not is_skip(first)
        assert is_skip(second)
        assert "same batch" in second

    def test_a_four_call_batch_still_costs_one_request(self, tool, xai):
        async def batch():
            for query in ("a", "b", "c", "d"):
                await tool.search_x(query, __message_id__="m2")

        arun(batch())
        assert xai.calls == 1

    def test_refusal_claims_nothing_about_what_x_contains(self, tool, xai):
        """An earlier version's wording led a model to report the gate as a finding."""

        async def batch():
            await tool.search_x("a", __message_id__="m3")
            return await tool.search_x("b", __message_id__="m3")

        refusal = arun(batch())
        assert "already cover" not in refusal
        assert "comprehensive" not in refusal

    def test_window_of_zero_disables_the_check(self, tool, xai):
        tool.valves.BATCH_WINDOW_SECONDS = 0

        async def batch():
            await tool.search_x("a", __message_id__="m4")
            return await tool.search_x("b", __message_id__="m4")

        assert not is_skip(arun(batch()))
        assert xai.calls == 2


class TestGenuineRetry:
    def test_retry_after_an_empty_result_is_allowed(self, tool, xai):
        """The failure that made an earlier, stricter gate unusable."""
        xai.queue(payload=EMPTY_PAYLOAD)

        async def scenario():
            first = await tool.search_x("today's top news", __message_id__="r1")
            await thinking()
            second = await tool.search_x("breaking news today", __message_id__="r1")
            return first, second

        first, second = arun(scenario())

        assert "No X posts were found" in first
        assert not is_skip(second)
        assert xai.calls == 2

    def test_retry_after_a_weak_result_is_allowed(self, tool, xai):
        async def scenario():
            await tool.search_x("q1", __message_id__="r2")
            await thinking()
            return await tool.search_x("q2 broader", __message_id__="r2")

        assert not is_skip(arun(scenario()))
        assert xai.calls == 2

    def test_a_later_turn_starts_fresh(self, tool, xai):
        async def scenario():
            await tool.search_x("q", __message_id__="turn-1")
            return await tool.search_x("another question", __message_id__="turn-2")

        assert not is_skip(arun(scenario()))
        assert xai.calls == 2


class TestBudget:
    def test_caps_runaway_retry_loops(self, tool, xai):
        async def scenario():
            results = []
            for index in range(4):
                results.append(await tool.search_x(f"q{index}", __message_id__="b1"))
                await thinking()
            return results

        results = arun(scenario())

        assert xai.calls == tool.valves.MAX_SEARCHES_PER_MESSAGE == 3
        assert not any(is_skip(result) for result in results[:3])
        assert is_skip(results[3])

    def test_limit_message_forbids_misreporting_it(self, tool, xai):
        tool.valves.MAX_SEARCHES_PER_MESSAGE = 1

        async def scenario():
            await tool.search_x("q1", __message_id__="b2")
            await thinking()
            return await tool.search_x("q2", __message_id__="b2")

        refusal = arun(scenario())
        assert "Do not describe this limit as evidence about X" in refusal
        assert "next turn" in refusal

    def test_zero_disables_gating_entirely(self, tool, xai):
        tool.valves.MAX_SEARCHES_PER_MESSAGE = 0

        async def scenario():
            for index in range(5):
                await tool.search_x(f"q{index}", __message_id__="b3")

        arun(scenario())
        assert xai.calls == 5


class TestDeduplication:
    def test_an_identical_repeat_is_served_from_memory(self, tool, xai):
        async def scenario():
            first = await tool.search_x("same query", __message_id__="d1")
            second = await tool.search_x("same query", __message_id__="d1")
            return first, second

        first, second = arun(scenario())

        assert first == second
        assert "Summary text." in second
        assert not is_skip(second)
        assert xai.calls == 1

    def test_differing_arguments_are_not_deduplicated(self, tool, xai):
        tool.valves.BATCH_WINDOW_SECONDS = 0

        async def scenario():
            await tool.search_x("q", __message_id__="d2")
            await tool.search_x("q", days=7, __message_id__="d2")

        arun(scenario())
        assert xai.calls == 2


class TestConcurrency:
    def test_truly_simultaneous_calls_are_refused(self, tool, xai, monkeypatch):
        """Belt and braces: correct even if a harness does run tools in parallel."""
        release = threading.Event()
        calls = []

        def blocking_post(url, headers=None, json=None, timeout=None):
            calls.append(json["input"][0]["content"])
            release.wait(timeout=5)

            class Response:
                status_code = 200

                def json(self):
                    return message_payload()

            return Response()

        monkeypatch.setattr(x_search.requests, "post", blocking_post)

        async def scenario():
            async def unblock():
                await asyncio.sleep(0.15)
                release.set()

            results = await asyncio.gather(
                tool.search_x("a", __message_id__="c1"),
                tool.search_x("b", __message_id__="c1"),
                unblock(),
            )
            return results[:2]

        results = arun(scenario())

        assert len(calls) == 1
        assert len([r for r in results if is_skip(r)]) == 1
        refusal = next(r for r in results if is_skip(r))
        assert "already running" in refusal


class TestFailuresReleaseTheBudget:
    def test_a_failed_search_does_not_consume_a_slot(self, tool, xai):
        xai.queue(raises=requests.ConnectionError("no route"))

        async def scenario():
            failed = await tool.search_x("q1", __message_id__="f1")
            await thinking()
            return failed, await tool.search_x("q2", __message_id__="f1")

        failed, retry = arun(scenario())

        assert failed.startswith("Error:")
        assert not is_skip(retry)

    def test_a_failed_search_leaves_no_phantom_in_flight_record(self, tool, xai):
        xai.queue(raises=requests.Timeout("slow"))

        async def scenario():
            await tool.search_x("q1", __message_id__="f2")
            # Immediately, with no pause: must not look like an in-flight sibling.
            return await tool.search_x("q2", __message_id__="f2")

        assert not is_skip(arun(scenario()))

    def test_http_errors_release_the_slot_too(self, tool, xai):
        xai.queue(payload={"error": {"message": "boom"}}, status=500)

        async def scenario():
            await tool.search_x("q1", __message_id__="f3")
            await thinking()
            return await tool.search_x("q2", __message_id__="f3")

        assert not is_skip(arun(scenario()))


class TestWithoutAMessageId:
    def test_gating_is_skipped_when_open_webui_gives_us_nothing(self, tool, xai):
        """Never break the tool because a harness did not supply the id."""

        async def scenario():
            for index in range(4):
                await tool.search_x(f"q{index}")

        arun(scenario())
        assert xai.calls == 4

    def test_metadata_message_id_is_used(self, tool, xai):
        async def scenario():
            await tool.search_x("q1", __metadata__={"message_id": "m"})
            return await tool.search_x("q2", __metadata__={"message_id": "m"})

        assert is_skip(arun(scenario()))
        assert xai.calls == 1


class TestHousekeeping:
    def test_stale_records_are_pruned(self, tool, xai):
        async def scenario():
            await tool.search_x("q", __message_id__="old")
            x_search._MESSAGE_CALLS["old"]["created"] -= x_search.GATE_TTL_SECONDS + 60
            await tool.search_x("q", __message_id__="new")

        arun(scenario())
        assert "old" not in x_search._MESSAGE_CALLS
        assert "new" in x_search._MESSAGE_CALLS
