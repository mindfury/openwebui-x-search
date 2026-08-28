# X Search for Open WebUI

An Open WebUI tool that gives any tool-calling model live access to X (formerly Twitter),
by delegating to xAI's server-side `x_search` on the Responses API.

Open WebUI ships web search, but nothing that searches X. The community hub has Grok
*model connections* — pipes and manifolds that let you chat with Grok — which is a
different thing: they don't give your existing models the ability to search X. This does.

Your chat model can be anything with working function calling. The Grok call happens
inside the tool.

## What this is not

It is easy to mistake this for a thin wrapper around the X API, and the distinction
matters more than the name suggests. This does **not** fetch a post by URL, take search
operators, or hand back raw posts and JSON. There is no cheap, fast, deterministic path
through it.

Every call starts a Grok agent that picks its own search queries, reads what it finds
across several rounds, and writes prose with citations. That makes it the right tool for
*"what are people saying about this"* and the wrong one for *"get me this post"* — and it
is why a call costs seconds and credits rather than milliseconds and nothing.

## Quick start

1. Copy all of [`x_search.py`](x_search.py) into **Workspace → Tools → Create**, and save.
2. Open the tool's **gear icon** and set `XAI_API_KEY` — get one at
   [console.x.ai](https://console.x.ai).
3. Turn it on in a chat with **+**, and ask something like *"what are people on X saying
   about the Grok 4.5 launch?"*

Every other setting has a working default. Two things to know before you rely on it:
each search takes **20–60 seconds**, and each one **spends xAI credits** separately from
your chat model. See [cost and latency](#cost-and-latency).

## What a call looks like

You ask a question; the tool hands it to a Grok agent that runs its own multi-step
investigation of X and comes back with a written answer plus the posts it used:

```
X search results for: "how are developers reacting to the Grok 4.5 launch?"
Scope: the last 7 days
Queries run on X: grok 4.5 developer reaction; grok 4.5 benchmarks criticism

Reaction is broadly positive on capability, with recurring complaints about
latency on long contexts [[1]](...) and pricing at the 200k tier [[2]](...).

Sources cited above:
[1] https://x.com/someone/status/111 (@someone)
[2] https://x.com/i/status/222 (X post)

Other sources consulted (not cited in the summary):
- https://x.ai/news
```

Sources also arrive as clickable citations on the message, and a status line ticks while
the search runs.

## Requirements

- Open WebUI with **Native** function calling (the default since v0.10.0 — Legacy mode's
  prompt-injected tool selection is unreliable)
- An [xAI API key](https://console.x.ai)
- A chat model with working tool calling

## Install

1. Copy the entire contents of [`x_search.py`](x_search.py).
2. In Open WebUI, go to **Workspace → Tools → Create**, paste, and **Save**.
3. Click the tool's **gear icon** and set `XAI_API_KEY`. It also falls back to an
   `XAI_API_KEY` environment variable on the server.
4. Enable it per chat with the **+** icon, or attach it to a model under
   **Workspace → Models → edit → Tools**.

Open WebUI parses the tool's docstrings into a spec when you hit **Save**, so any later
edit needs another paste and save to take effect.

## The tool

A single tool, `search_x`. Only `query` is required.

| Parameter | Description |
|---|---|
| `query` | The question to research, phrased as you'd put it to a person |
| `days` | Restrict to the last N days. `0` (default) means no recency limit |
| `from_date` / `to_date` | Explicit `YYYY-MM-DD` window. Overrides `days` |
| `allowed_handles` | Only these handles, max 20. Not combinable with exclusions |
| `excluded_handles` | Leave these out, max 20. Not combinable with `allowed_handles` |

Handles are accepted as `@name`, `name`, or a profile/post URL. URLs built on a
reserved X path carry no handle (`x.com/i/status/…`, `x.com/home`) and are rejected
rather than read as an account called `i` or `home`. If a handle filter is asked for
but nothing in it parses, the search is refused rather than run unfiltered — running
it would be broader than what was asked for, and you'd pay for a full investigation
to find out.

## Cost and latency

Worth understanding before you attach this to a busy model:

- **A call takes 20–60 seconds.** The search agent runs several X searches internally
  before answering. The timeout defaults to 180s.
- **Each call spends xAI credits**, separately from whatever model you're chatting with.
  `grok-4.5` is the priciest option; `MODEL` can be pointed elsewhere, though xAI
  publishes no per-model capability matrix for `x_search` and all their examples use
  `grok-4.5`.
- **Image and video understanding are off by default.** They analyse media attached to
  matched posts, and they are the single biggest cost and latency lever here.

## The call gate

Models routinely split one question into several tool calls with differently worded
queries, issued together before any results come back. That's good practice against a
cheap keyword API and wasteful here, because `x_search` already fans out across many
phrasings internally — so you pay for several full investigations covering the same
ground, and wait for them one after another.

The tool tracks calls per assistant message and:

- **refuses fan-out siblings** — a call arriving within `BATCH_WINDOW_SECONDS` of the
  previous one finishing was issued in the same batch, before any result could have been
  read;
- **allows genuine retries** — once the model has read a result and decided it fell
  short, searching again is the right call, and a full inference pass takes seconds;
- **serves identical repeats from memory**, costing nothing;
- **caps runaway loops** at `MAX_SEARCHES_PER_MESSAGE`.

Refusals return a sentence telling the model to answer from the results it already has.
They deliberately claim nothing about what X contains, so a refusal can't be mistaken for
a finding.

Set either threshold to `0` to disable that half.

## Configuration

All valves live under the tool's gear icon.

| Valve | Default | Purpose |
|---|---|---|
| `XAI_API_KEY` | env `XAI_API_KEY` | xAI credentials |
| `BASE_URL` | `https://api.x.ai/v1` | API base |
| `MODEL` | `grok-4.5` | Must support the `x_search` server-side tool |
| `INSTRUCTIONS` | see file | System instructions for the search agent |
| `DEFAULT_EXCLUDED_HANDLES` | — | Always excluded, unless a call restricts to handles |
| `ENABLE_IMAGE_UNDERSTANDING` | `false` | Analyse images in posts |
| `ENABLE_VIDEO_UNDERSTANDING` | `false` | Analyse videos in posts |
| `MAX_OUTPUT_TOKENS` | `0` | `0` leaves it to the API |
| `TIMEOUT_SECONDS` | `180` | HTTP timeout |
| `MAX_SEARCHES_PER_MESSAGE` | `3` | Runaway-loop ceiling; `0` disables |
| `BATCH_WINDOW_SECONDS` | `2.0` | Fan-out detection window; `0` disables |
| `EXTRA_TOOL_PARAMS` | — | JSON merged into the `x_search` config |
| `EMIT_STATUS` / `EMIT_CITATIONS` | `true` | Chat UI events |

Per-user valves (`ALLOWED_HANDLES`, `EXCLUDED_HANDLES`, `ENABLE_MEDIA_UNDERSTANDING`)
apply to every search that user runs.

A stored valve value always beats a new default in the code, so after updating the
script, check the gear icon if a changed default doesn't seem to have taken.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

The suite covers request building, valve plumbing, response parsing, every error path,
and the gate — including a genuinely concurrent case using a blocking transport. It
never touches the network.

## Limitations

- Gate state is per-process and in memory. Across multiple Open WebUI workers, two calls
  landing on different workers could each get a slot — no worse than having no gate.
- The gate distinguishes fan-out from retry by timing, because the tool-calling protocol
  passes no batch identity down to the tool. It's a heuristic, though a well-separated
  one: batch siblings arrive milliseconds apart, retries take seconds.
- Broad queries like "today's top news" return little; X search works better on a
  specific topic.
- `MAX_OUTPUT_TOKENS` caps the search agent mid-answer rather than shortening it. If
  it's set too low the response comes back truncated, and the tool reports that as an
  error naming the valve — it will not pass it off as "nothing found on X".

## License

MIT — see [LICENSE](LICENSE).

Built by [mindfury](https://github.com/mindfury). Bug reports and feature requests are
welcome as GitHub issues.
