# Publishing to the Open WebUI Community Hub

Paste-ready copy for the listing at [openwebui.com](https://openwebui.com). The hub takes
the tool's name and description from the docstring header in `x_search.py`, so those are
already written for a cold reader — this file is the surrounding copy.

Keep the version in the docstring header and [CHANGELOG.md](CHANGELOG.md) in step with
whatever you publish.

---

## Name

```
X Search
```

## Short description

Taken automatically from the `description:` line in the docstring header:

```
Give any model live access to X (Twitter). A Grok agent researches your question on X
and returns a written answer citing the posts it used. Needs an xAI API key; each
search takes 20-60s and spends xAI credits.
```

## Long description

```markdown
Open WebUI can search the web, but not X. This tool adds it, by handing your question
to xAI's server-side `x_search` — a Grok agent that runs its own multi-step
investigation of X and comes back with a written answer plus the posts it used.

It works with **any** chat model that supports tool calling. You do not have to be
using Grok; the Grok call happens inside the tool.

### This is not an X API wrapper

Worth being clear about, because the name invites the wrong assumption. It does not
fetch a post by URL, take search operators, or return raw posts and JSON. Every call
starts a Grok agent that chooses its own queries, reads across several rounds of
results, and writes a cited answer. Right tool for *"what are people saying about
this"*, wrong tool for *"get me this post"* — and that is why a call costs seconds
and credits instead of milliseconds and nothing.

**Good for:** reaction to news, sentiment about a person or product, what an account
has been posting, trending topics, breaking events where first-hand posts beat
articles. It can also find accounts worth following on a subject, or pull a full
thread from a post URL.

### Setup

1. Paste the tool in, save it.
2. Open the gear icon and set `XAI_API_KEY` — get one at https://console.x.ai.
3. Enable it in a chat with **+**, or attach it to a model under
   Workspace → Models → edit → Tools.

Every other setting has a working default.

### Before you install

- **Each search takes 20-60 seconds.** The agent runs several searches on X before
  answering. This is not a fast keyword lookup.
- **Each search spends xAI credits**, separately from your chat model.
- Image and video understanding are off by default. They are the biggest cost and
  latency lever if you turn them on.

### Notes

Results carry the search agent's own citation numbering, so a `[2]` in the summary
points at the post listed as `[2]`, and sources it merely consulted are listed
separately from ones it cited.

The tool also limits itself to one search per assistant message when a model fires
several at once. Models habitually split a question into parallel searches, which is
sound against a cheap keyword API but wasteful here — `x_search` already searches many
phrasings internally, so the extra calls buy duplicate work at full price and double
the wait. Genuine follow-up searches, after the model has read a result, still go
through. Both thresholds are configurable, and can be switched off.

Requires Open WebUI with Native function calling (the default since v0.10.0).

Source, full documentation, and tests: https://github.com/mindfury/openwebui-x-search
MIT licensed.
```

## Suggested tags

```
x, twitter, search, xai, grok, social-media, research, news, real-time, citations
```

## Before publishing, check

- [ ] `version:` in the docstring header matches [CHANGELOG.md](CHANGELOG.md)
- [ ] Tests pass (`pytest`)
- [ ] No API key is pasted into the tool body — it belongs in the valve
- [ ] `author_url` points at the repo
