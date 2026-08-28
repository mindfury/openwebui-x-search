"""
title: X Search
description: Give any model live access to X (Twitter). Not an API wrapper that fetches posts - a Grok agent researches your question on X and writes a cited answer. Needs an xAI API key; 20-60s and xAI credits per search.
author: mindfury
author_url: https://github.com/mindfury/openwebui-x-search
version: 1.0.1
license: MIT
requirements: requests
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

import requests
from pydantic import BaseModel, Field

# xAI caps allowed_x_handles / excluded_x_handles at 20 entries each.
MAX_HANDLES = 20

# Searches already run while generating a given assistant message, so that a model
# fanning one question out into parallel tool calls does not pay for the same work
# several times over. Keyed by message id; entries are pruned by age.
_MESSAGE_CALLS: dict[str, dict] = {}
GATE_TTL_SECONDS = 900

# Matches a bare handle, an @handle, or a profile/post URL we can pull a handle out of.
# The URL form requires the handle segment to end at a delimiter, so an over-long
# segment is rejected outright rather than silently truncated to its first 15
# characters — which would yield a wrong but entirely plausible handle.
HANDLE_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:x|twitter)\.com/(?:#!/)?(?P<url>[A-Za-z0-9_]{1,15})(?=[/?#]|$)"
    r"|^@?(?P<bare>[A-Za-z0-9_]{1,15})$",
    re.IGNORECASE,
)

# First path segments X reserves for itself, so a URL built on one carries no handle.
# xAI's own citations routinely use the handle-less x.com/i/status/<id> form.
RESERVED_X_PATHS = frozenset(
    {
        "compose",
        "explore",
        "hashtag",
        "home",
        "i",
        "intent",
        "login",
        "logout",
        "messages",
        "notifications",
        "search",
        "settings",
        "share",
        "signup",
        "topics",
    }
)

DEFAULT_INSTRUCTIONS = (
    "You are a research assistant with live access to X (formerly Twitter). "
    "Answer using X posts only. Summarise what is actually being posted, and "
    "quote the most relevant posts verbatim (short excerpts) with the author's "
    "@handle and the post date. Cover the range of viewpoints if opinions differ, "
    "and say plainly when there is little or no relevant discussion rather than "
    "padding the answer. Do not speculate beyond what the posts say."
)


def _normalise_handles(raw: str) -> tuple[list[str], Optional[str]]:
    """Turn a comma/space separated blob into a clean handle list.

    Returns (handles, warning).
    """
    if not raw:
        return [], None

    handles: list[str] = []
    rejected: list[str] = []
    reserved: list[str] = []
    for chunk in re.split(r"[,\s]+", raw.strip()):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = HANDLE_RE.match(chunk)
        if not match:
            rejected.append(chunk)
            continue
        handle = match.group("url") or match.group("bare")
        # Only the URL form can name a reserved path; a bare "home" is far more
        # likely to be someone naming the @home account than a mistake.
        if match.group("url") and handle.lower() in RESERVED_X_PATHS:
            reserved.append(chunk)
            continue
        if handle.lower() not in {h.lower() for h in handles}:
            handles.append(handle)

    warnings: list[str] = []
    if rejected:
        warnings.append(f"Ignored unparseable handle(s): {', '.join(rejected)}.")
    if reserved:
        warnings.append(f"Ignored URL(s) that carry no handle: {', '.join(reserved)}.")
    if len(handles) > MAX_HANDLES:
        dropped = handles[MAX_HANDLES:]
        handles = handles[:MAX_HANDLES]
        warnings.append(
            f"Only the first {MAX_HANDLES} handles were used (dropped: {', '.join(dropped)})."
        )

    return handles, " ".join(warnings) or None


def _prune_message_calls(now: float) -> None:
    """Drop gate records for messages that finished generating a while ago."""
    for key in [
        key
        for key, entry in _MESSAGE_CALLS.items()
        if now - entry["created"] > GATE_TTL_SECONDS
    ]:
        _MESSAGE_CALLS.pop(key, None)


def _resolve_message_id(message_id: Any, metadata: Any) -> Optional[str]:
    """The assistant message currently being generated, if Open WebUI told us."""
    if isinstance(message_id, str) and message_id:
        return message_id
    if isinstance(metadata, dict):
        for key in ("message_id", "assistant_message_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _validate_date(value: str, label: str) -> tuple[str, Optional[str]]:
    """Validate an ISO8601 YYYY-MM-DD date. Returns (value, error)."""
    if not value:
        return "", None
    value = value.strip()
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return "", f"`{label}` must be in YYYY-MM-DD format, got '{value}'."
    return value, None


def _iter_content_blocks(payload: dict) -> Any:
    """Yield every content block across every output item."""
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict):
                yield block


def _extract_text(payload: dict) -> str:
    """Pull the assistant's answer out of a Responses API payload."""
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("output_text", "text"):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    return "\n\n".join(parts).strip()


def _incomplete_reason(payload: dict) -> Optional[str]:
    """Why the API stopped early, if it did.

    A response truncated part-way carries no message block, so without this it is
    indistinguishable from a search that genuinely found nothing — and would be
    reported to the model as a fact about X rather than a limit on our side.
    """
    if payload.get("status") != "incomplete":
        return None
    details = payload.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    return str(reason) if reason else "reason unspecified"


def _source_name(url: str) -> str:
    """A readable label for a cited URL, e.g. '@elonmusk' for an X post.

    Note that xAI also returns handle-less forms like x.com/i/status/<id> and
    x.com/i/user/<id>; '/i/' is a reserved path, not a handle.
    """
    x_path = re.match(
        r"^https?://(?:www\.)?(?:x|twitter)\.com/(?P<path>.*)$", url, re.IGNORECASE
    )
    if not x_path:
        domain = re.sub(r"^https?://(?:www\.)?", "", url, flags=re.IGNORECASE).split("/")[0]
        return domain or url

    segments = [s for s in x_path.group("path").split("/") if s]
    if (
        segments
        and segments[0].lower() not in RESERVED_X_PATHS
        and re.fullmatch(r"[A-Za-z0-9_]{1,15}", segments[0])
    ):
        return f"@{segments[0]}"
    return "X post" if "status" in segments else "X profile"


def _extract_citations(payload: dict) -> list[dict]:
    """Collect cited sources, keeping the model's own citation numbering.

    Inline annotations carry the number the summary text refers to ("[[2]](url)"),
    so those are returned first and in that order. The top-level `citations` field
    is every source the agent *encountered*, cited or not, so anything left over is
    returned after them and marked as uncited.

    Returns [{"url", "label", "cited"}].
    """
    cited: dict[str, str] = {}

    for block in _iter_content_blocks(payload):
        for annotation in block.get("annotations") or []:
            if not isinstance(annotation, dict):
                continue
            url = annotation.get("url")
            if not isinstance(url, str) or not url.startswith("http"):
                continue
            label = annotation.get("title")
            label = label.strip().strip("[]") if isinstance(label, str) else ""
            if url not in cited or (label and not cited[url]):
                cited[url] = label

    # Keep the model's numbering only when *every* annotation gave us a number.
    # `title` is a page title in the general Responses API shape and only sometimes
    # the bracket number the summary refers to, so this is all or nothing: a partial
    # set would print page titles where numbers belong, or collide with the
    # positional fallback used for the annotations that carry no number.
    ordered = list(cited.items())
    if ordered and all(label.isdigit() for _, label in ordered):
        ordered.sort(key=lambda item: int(item[1]))
    else:
        ordered = [(url, "") for url, _ in ordered]

    results = [{"url": url, "label": label, "cited": True} for url, label in ordered]

    for citation in payload.get("citations") or []:
        url = citation if isinstance(citation, str) else None
        if isinstance(citation, dict):
            url = citation.get("url") or citation.get("source")
        if isinstance(url, str) and url.startswith("http") and url not in cited:
            cited[url] = ""
            results.append({"url": url, "label": "", "cited": False})

    return results


def _extract_searches(payload: dict) -> list[str]:
    """Describe the searches the server-side tool actually ran, for status display."""
    searches: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type") or ""
        if "search" not in item_type or not item_type.endswith("_call"):
            continue

        query = None
        action = item.get("action")
        if isinstance(action, dict):
            query = action.get("query") or action.get("q")
        elif isinstance(action, str):
            query = action
        if not query:
            query = item.get("query")
        if not query:
            arguments = item.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                    if isinstance(parsed, dict):
                        query = parsed.get("query") or parsed.get("q")
                except json.JSONDecodeError:
                    pass

        if isinstance(query, str) and query.strip() and query not in searches:
            searches.append(query.strip())
    return searches


def _error_detail(response: requests.Response) -> str:
    """Best-effort human readable message from an xAI error response."""
    try:
        body = response.json()
    except ValueError:
        return (response.text or "").strip()[:500]

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
        if body.get("message"):
            return str(body["message"])
    return json.dumps(body)[:500]


async def _perform_search(
    valves: "Tools.Valves",
    query: str,
    from_date: str,
    to_date: str,
    allowed_handles: str,
    excluded_handles: str,
    window_note: str,
    event_emitter: Optional[Callable[[dict], Awaitable[None]]],
    user: Optional[dict],
    message_id: Optional[str] = None,
) -> str:
    """Everything behind the tool method: gating, the API call, and result formatting."""

    async def status(description: str, done: bool = False) -> None:
        if event_emitter and valves.EMIT_STATUS:
            await event_emitter(
                {"type": "status", "data": {"description": description, "done": done}}
            )

    if not isinstance(query, str) or not query.strip():
        return "Error: `query` is required."
    query = query.strip()

    api_key = (valves.XAI_API_KEY or "").strip()
    if not api_key:
        return (
            "Error: no xAI API key configured. Set the XAI_API_KEY valve on this tool "
            "(Workspace → Tools → X Search → gear icon), or set the XAI_API_KEY "
            "environment variable on the Open WebUI server."
        )

    user_valves = (user or {}).get("valves")
    notes: list[str] = []

    # The API rejects both filters in the same request. Checked on the raw arguments,
    # before parsing, so that an unparseable exclusion cannot quietly swallow the
    # conflict and leave the search running with only half of what was asked for.
    if (allowed_handles or "").strip() and (excluded_handles or "").strip():
        return (
            "Error: `allowed_handles` and `excluded_handles` cannot be used in the "
            "same search. Pick one."
        )

    allowed_raw = allowed_handles or getattr(user_valves, "ALLOWED_HANDLES", "") or ""
    excluded_raw = (
        excluded_handles
        or getattr(user_valves, "EXCLUDED_HANDLES", "")
        or valves.DEFAULT_EXCLUDED_HANDLES
        or ""
    )
    excluded_from_call = bool((excluded_handles or "").strip())

    allowed, allowed_warning = _normalise_handles(allowed_raw)
    excluded, excluded_warning = _normalise_handles(excluded_raw)

    # A filter that was asked for but parsed to nothing would search all of X: broader
    # than what was asked for, and a full agent investigation spent before anyone finds
    # out. An allow-list failing open is a scope violation, so it is refused whatever
    # its source. A call-level exclusion is refused too. Only the instance-wide
    # exclusion default degrades to a note, so one bad valve cannot brick every search.
    if allowed_raw.strip() and not allowed:
        detail = f" {allowed_warning}" if allowed_warning else ""
        return (
            f"Error: no usable X handle in the allowed-handle filter.{detail} Searching "
            "without the restriction would be broader than asked, so nothing was sent."
        )
    if excluded_from_call and not excluded:
        detail = f" {excluded_warning}" if excluded_warning else ""
        return (
            f"Error: no usable X handle in `excluded_handles`.{detail} Searching without "
            "the exclusion would be broader than asked, so nothing was sent."
        )

    if allowed_warning:
        notes.append(allowed_warning)

    if allowed and excluded:
        excluded = []
        excluded_warning = None
        notes.append(
            ("Call-level" if excluded_from_call else "Default")
            + " excluded handles were ignored because the search is restricted to "
            "specific handles."
        )
    if excluded_warning:
        notes.append(excluded_warning)

    from_date, error = _validate_date(from_date, "from_date")
    if error:
        return f"Error: {error}"
    to_date, error = _validate_date(to_date, "to_date")
    if error:
        return f"Error: {error}"
    if from_date and to_date and from_date > to_date:
        return f"Error: `from_date` ({from_date}) is after `to_date` ({to_date})."

    media = valves.ENABLE_IMAGE_UNDERSTANDING or getattr(
        user_valves, "ENABLE_MEDIA_UNDERSTANDING", False
    )
    video = valves.ENABLE_VIDEO_UNDERSTANDING or getattr(
        user_valves, "ENABLE_MEDIA_UNDERSTANDING", False
    )

    tool_config: dict[str, Any] = {"type": "x_search"}
    if allowed:
        tool_config["allowed_x_handles"] = allowed
    elif excluded:
        tool_config["excluded_x_handles"] = excluded
    if from_date:
        tool_config["from_date"] = from_date
    if to_date:
        tool_config["to_date"] = to_date
    if media:
        tool_config["enable_image_understanding"] = True
    if video:
        tool_config["enable_video_understanding"] = True

    if valves.EXTRA_TOOL_PARAMS.strip():
        try:
            extra = json.loads(valves.EXTRA_TOOL_PARAMS)
            if not isinstance(extra, dict):
                raise ValueError("must be a JSON object")
            tool_config.update(extra)
        except (json.JSONDecodeError, ValueError) as exc:
            return f"Error: EXTRA_TOOL_PARAMS is not a valid JSON object ({exc})."

    body: dict[str, Any] = {
        "model": valves.MODEL,
        "input": [{"role": "user", "content": query}],
        "tools": [tool_config],
    }
    if valves.INSTRUCTIONS.strip():
        body["instructions"] = valves.INSTRUCTIONS.strip()
    if valves.MAX_OUTPUT_TOKENS > 0:
        body["max_output_tokens"] = valves.MAX_OUTPUT_TOKENS

    # Models routinely fan one question out into several *parallel* tool calls with
    # differently worded queries, before any results exist to justify them. That
    # duplicates the fan-out x_search already does internally, at full price each.
    # A *sequential* repeat is different: the model has read the first result and
    # decided it fell short, which is a legitimate retry and must be allowed through.
    # So concurrent calls are refused, sequential ones are allowed up to a budget.
    # Both are keyed to the assistant message being generated, leaving follow-up
    # questions on later turns unaffected.
    record: Optional[dict] = None
    entry: Optional[dict] = None
    if message_id and valves.MAX_SEARCHES_PER_MESSAGE > 0:
        now = time.time()
        _prune_message_calls(now)
        entry = _MESSAGE_CALLS.setdefault(message_id, {"created": now, "calls": []})
        signature = json.dumps(
            {"model": valves.MODEL, "query": query, "tool": tool_config},
            sort_keys=True,
            default=str,
        )

        for previous in entry["calls"]:
            if previous["signature"] == signature and not previous["pending"]:
                await status("Reused an identical search from this message.", done=True)
                return previous["result"]

        in_flight = [call for call in entry["calls"] if call["pending"]]
        if in_flight:
            running = "; ".join(f'"{call["query"]}"' for call in in_flight)
            await status("Skipped a simultaneous X search.", done=True)
            return (
                f"Skipped: an X search is already running for this message ({running}), so "
                "this one was not sent. Its results arrive in this same turn — read them "
                "first. If they genuinely fall short you may search again afterwards; do not "
                "issue searches in parallel before seeing any results."
            )

        # Open WebUI runs a model's parallel tool calls serially, so a fan-out batch
        # never actually overlaps in time. It is still recognisable: the harness starts
        # the next call the instant the previous returns, with no model inference in
        # between, whereas a genuine retry needs a full inference pass first. So a call
        # arriving within a second or two of the last one finishing was issued blind,
        # as part of the same batch, before any results could have been read.
        window = valves.BATCH_WINDOW_SECONDS
        finished = [call["finished"] for call in entry["calls"] if call["finished"]]
        if window > 0 and finished and (now - max(finished)) < window:
            batch = "; ".join(f'"{call["query"]}"' for call in entry["calls"])
            await status("Skipped a same-batch X search.", done=True)
            return (
                f"Skipped: this search was issued in the same batch as {batch}, before those "
                "results could have been read, so it was not sent. Those results are in this "
                "same turn — read them and answer from them. If they genuinely leave a gap, "
                "search once more after reading them, not alongside them."
            )

        if len(entry["calls"]) >= valves.MAX_SEARCHES_PER_MESSAGE:
            already = "; ".join(f'"{call["query"]}"' for call in entry["calls"])
            await status("X search limit reached for this message.", done=True)
            return (
                f"Skipped: this message has already run {len(entry['calls'])} X searches "
                f"({already}) and the per-message limit is reached. Answer with what those "
                "returned — including saying plainly that little was found, if that is the "
                "case. Do not describe this limit as evidence about X. You can search again "
                "on the next turn."
            )

        # Reserve the slot before awaiting, so calls issued in parallel see it.
        record = {
            "signature": signature,
            "query": query,
            "result": "",
            "pending": True,
            "finished": None,
        }
        entry["calls"].append(record)

    def release_slot() -> None:
        """Give the budget back when the search failed and produced nothing."""
        if record is not None and entry is not None and record in entry["calls"]:
            entry["calls"].remove(record)

    def complete(text: str) -> str:
        """Mark this search finished so a later call in the same message can proceed."""
        if record is not None:
            record["result"] = text
            record["pending"] = False
            record["finished"] = time.time()
        return text

    scope = []
    if window_note:
        scope.append(window_note)
    elif from_date or to_date:
        scope.append(f"{from_date or 'earliest'} → {to_date or 'now'}")
    if allowed:
        scope.append("@" + ", @".join(allowed))
    scope_text = "; ".join(scope)
    await status(
        f"Searching X for “{query}”" + (f" ({scope_text})" if scope_text else "") + "…"
    )

    url = f"{valves.BASE_URL.rstrip('/')}/responses"
    try:
        response = await asyncio.to_thread(
            requests.post,
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=valves.TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        release_slot()
        await status("X search timed out.", done=True)
        return (
            f"Error: the X search timed out after {valves.TIMEOUT_SECONDS}s. "
            "Try a narrower query or raise the TIMEOUT_SECONDS valve."
        )
    except requests.RequestException as exc:
        release_slot()
        await status("X search failed.", done=True)
        return f"Error: could not reach the xAI API ({exc})."

    if response.status_code != 200:
        detail = _error_detail(response)
        release_slot()
        await status("X search failed.", done=True)
        if response.status_code in (401, 403):
            return f"Error: xAI rejected the API key ({response.status_code}). {detail}"
        if response.status_code == 429:
            return f"Error: xAI rate limit or quota reached (429). {detail}"
        if response.status_code == 404:
            return (
                f"Error: model '{valves.MODEL}' or endpoint not found (404). {detail} "
                "Check the MODEL valve names a Grok model that supports x_search."
            )
        return f"Error: xAI API returned {response.status_code}. {detail}"

    try:
        payload = response.json()
    except ValueError:
        payload = None
    # Valid JSON that is not an object would crash every reader below on .get().
    if not isinstance(payload, dict):
        release_slot()
        await status("X search failed.", done=True)
        return "Error: the xAI API returned a response that was not valid JSON."

    text = _extract_text(payload)
    citations = _extract_citations(payload)
    searches = _extract_searches(payload)

    # Distinguish "the search was cut short" from "X had nothing", which look identical
    # in the payload. The slot is not released: the call was billed, and a retry would
    # hit the same cap rather than fixing anything.
    incomplete = _incomplete_reason(payload)
    if incomplete and not text:
        await status("X search was cut short.", done=True)
        hint = (
            " Raise the MAX_OUTPUT_TOKENS valve, or set it to 0 to leave the cap to the API."
            if incomplete == "max_output_tokens"
            else ""
        )
        return complete(
            f"Error: the search model stopped before writing an answer ({incomplete}), so "
            f"no results came back for “{query}”. This is a limit on this end, not a "
            f"finding about X — do not report it as one.{hint}"
        )
    if incomplete:
        notes.append(
            f"The search model's answer was cut short ({incomplete}) and may be unfinished."
        )

    if not text and not citations:
        await status("No X posts found.", done=True)
        hint = (
            " Try widening the date range or the handle filter."
            if (from_date or to_date or allowed)
            else ""
        )
        return complete(f"No X posts were found for “{query}”.{hint}")

    if citations and event_emitter and valves.EMIT_CITATIONS:
        for citation in citations:
            name = _source_name(citation["url"])
            await event_emitter(
                {
                    "type": "citation",
                    "data": {
                        "document": [citation["url"]],
                        "metadata": [
                            {
                                "source": citation["url"],
                                "date_accessed": datetime.now(timezone.utc).isoformat(),
                            }
                        ],
                        "source": {
                            "name": name if citation["cited"] else f"{name} (consulted)",
                            "url": citation["url"],
                        },
                    },
                }
            )

    count = sum(1 for citation in citations if citation["cited"])
    await status(
        f"Found {count} X {'source' if count == 1 else 'sources'}."
        if count
        else "X search complete.",
        done=True,
    )

    lines = [f'X search results for: "{query}"']
    if scope_text:
        lines.append(f"Scope: {scope_text}")
    if searches:
        lines.append("Queries run on X: " + "; ".join(searches))
    lines.append("")
    lines.append(text or "(The model returned citations but no summary text.)")

    # Numbering here deliberately reuses the model's own citation numbers so that a
    # "[2]" in the summary above points at entry [2] below.
    used = [c for c in citations if c["cited"]]
    if used:
        lines.append("")
        lines.append("Sources cited above:")
        for index, citation in enumerate(used, start=1):
            number = citation["label"] or str(index)
            lines.append(f"[{number}] {citation['url']} ({_source_name(citation['url'])})")

    consulted = [c for c in citations if not c["cited"]]
    if consulted:
        lines.append("")
        lines.append("Other sources consulted (not cited in the summary):")
        for citation in consulted:
            lines.append(f"- {citation['url']}")

    if notes:
        lines.append("")
        lines.append("Notes: " + " ".join(notes))

    # Completing also lets an identical repeat in this message reuse the result for free.
    return complete("\n".join(lines))


class Tools:
    class Valves(BaseModel):
        XAI_API_KEY: str = Field(
            default=os.getenv("XAI_API_KEY", ""),
            description="Required — the tool does nothing until this is set. Create a key at https://console.x.ai. Falls back to an XAI_API_KEY environment variable on the Open WebUI server. Every other valve below can be left alone.",
        )
        BASE_URL: str = Field(
            default="https://api.x.ai/v1",
            description="xAI API base URL.",
        )
        MODEL: str = Field(
            default="grok-4.5",
            description="Model used to run the search. Must be a Grok model that supports the server-side x_search tool.",
        )
        INSTRUCTIONS: str = Field(
            default=DEFAULT_INSTRUCTIONS,
            description="System instructions given to the search model. Leave empty to send none.",
        )
        DEFAULT_EXCLUDED_HANDLES: str = Field(
            default="",
            description="Comma separated X handles to always exclude (e.g. spam accounts). Ignored when a call specifies allowed handles.",
        )
        ENABLE_IMAGE_UNDERSTANDING: bool = Field(
            default=False,
            description="Let the search model analyse images attached to posts. Slower and more expensive.",
        )
        ENABLE_VIDEO_UNDERSTANDING: bool = Field(
            default=False,
            description="Let the search model analyse videos attached to posts. Slower and more expensive.",
        )
        MAX_OUTPUT_TOKENS: int = Field(
            default=0,
            description="Cap on the search model's output tokens. 0 leaves it to the API.",
        )
        TIMEOUT_SECONDS: int = Field(
            default=180,
            description="HTTP timeout. X Search is agentic and can take a while on broad queries.",
        )
        MAX_SEARCHES_PER_MESSAGE: int = Field(
            default=3,
            description=(
                "Ceiling on how many xAI searches one assistant message may run, as a backstop "
                "against runaway retry loops. Separately, and regardless of this number, a search "
                "issued while another is still running is always refused: that is the parallel "
                "fan-out models do before any results exist, and it duplicates the fan-out "
                "x_search already does internally. Sequential retries after a weak result are "
                "allowed, since by then the model has seen the results. 0 disables both checks."
            ),
        )
        BATCH_WINDOW_SECONDS: float = Field(
            default=2.0,
            description=(
                "Seconds after a search finishes during which another search for the same "
                "message is treated as part of the same fan-out batch and refused. Open WebUI "
                "runs a model's parallel tool calls back to back, so a batch sibling arrives "
                "almost instantly, whereas a genuine retry needs a full model inference first "
                "and takes seconds. Raise it if fan-out still gets through, lower it if real "
                "retries are being blocked. 0 disables this check."
            ),
        )
        EXTRA_TOOL_PARAMS: str = Field(
            default="",
            description='Optional JSON object merged into the x_search tool config, for parameters this tool does not expose yet, e.g. {"max_search_results": 30}.',
        )
        EMIT_STATUS: bool = Field(
            default=True,
            description="Show progress messages in the chat while searching.",
        )
        EMIT_CITATIONS: bool = Field(
            default=True,
            description="Attach the cited X posts as clickable citations on the message.",
        )

    class UserValves(BaseModel):
        ALLOWED_HANDLES: str = Field(
            default="",
            description="Comma separated X handles to restrict every search to (max 20). Cannot be combined with excluded handles.",
        )
        EXCLUDED_HANDLES: str = Field(
            default="",
            description="Comma separated X handles to exclude from every search (max 20).",
        )
        ENABLE_MEDIA_UNDERSTANDING: bool = Field(
            default=False,
            description="Also analyse images and videos attached to posts.",
        )

    def __init__(self):
        self.valves = self.Valves()
        # Let this tool emit its own citations (the real X post URLs) instead of
        # Open WebUI citing the raw tool output.
        self.citation = False

    async def search_x(
        self,
        query: str,
        days: int = 0,
        from_date: str = "",
        to_date: str = "",
        allowed_handles: str = "",
        excluded_handles: str = "",
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __user__: Optional[dict] = None,
        __message_id__: Optional[str] = None,
        __metadata__: Optional[dict] = None,
    ) -> str:
        """
        Commission a research agent to investigate a question on X (formerly Twitter).

        This is not a search index. It is a live agent that spends 20-60 seconds and real
        API budget running its own multi-step investigation of X — keyword, semantic and
        account searches across many phrasings it chooses itself — and returns a written
        answer citing the posts it used. Brief it the way you would brief a researcher:
        one substantial question covering everything you want to know about the topic,
        phrased as you would to a person.

        There is nothing to gain by issuing more than one call for a topic. A recent view
        and a background view are this same call with a different `days` value, and running
        both wastes a full agent investigation for material the broader one already covers.
        The multi-angle fan-out you might do by hand is what the agent already does
        internally. Ask once, read the answer, and only follow up if it left a real gap.

        Use it for what people are actually posting: reaction to news, sentiment about a
        person or product, what an account has been saying, trending topics, or breaking
        events where first-hand posts beat articles. It can also find accounts worth
        following on a subject, or pull a full thread given a post URL. Prefer it over a
        general web search for anything about X, Twitter, tweets, or "what people are
        saying".

        :param query: The question to research, phrased as you would to a person, e.g. "how are developers reacting to the Grok 4.5 launch, and what are the main criticisms?".
        :param days: Restrict to the last N days, counting today, for "latest" / "today" / "this week" questions. It works the dates out for you, so you do not need to know today's date. Leave as 0 for no recency limit.
        :param from_date: Optional earliest post date, ISO8601 YYYY-MM-DD, for a specific historical window. Overrides days.
        :param to_date: Optional latest post date, ISO8601 YYYY-MM-DD. Overrides days.
        :param allowed_handles: Optional comma separated X handles to search within, max 20, e.g. "elonmusk, xai". Cannot be combined with excluded_handles.
        :param excluded_handles: Optional comma separated X handles to leave out, max 20. Cannot be combined with allowed_handles.
        """
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 0

        # An explicit date range is more specific, so it wins; the Scope line in the
        # result always reports the window actually used.
        window_note = ""
        if days > 0 and not from_date and not to_date:
            days = min(days, 365)
            today = datetime.now(timezone.utc).date()
            from_date = (today - timedelta(days=days - 1)).isoformat()
            to_date = today.isoformat()
            window_note = "today" if days == 1 else f"the last {days} days"

        return await _perform_search(
            valves=self.valves,
            query=query,
            from_date=from_date,
            to_date=to_date,
            allowed_handles=allowed_handles,
            excluded_handles=excluded_handles,
            window_note=window_note,
            event_emitter=__event_emitter__,
            user=__user__,
            message_id=_resolve_message_id(__message_id__, __metadata__),
        )
