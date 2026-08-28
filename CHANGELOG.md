# Changelog

## 1.0.1

Fixes found in a review of 1.0.0.

- Citation numbering is now all-or-nothing. `annotation.title` is a page title in the
  general Responses API shape and only sometimes the bracket number the summary refers
  to; reusing one as a number printed titles where numbers belonged, and a partial set
  collided with the positional fallback. Either every annotation supplies a number or
  the list is numbered sequentially.
- A response the API cut short is no longer reported as "No X posts were found". It
  carries no message block, so it was indistinguishable from an empty result — and the
  tool was stating a falsehood about X on the strength of a local token cap. It now
  returns an error naming the cause, and flags a partial answer as unfinished.
- URLs built on a reserved X path (`x.com/i/status/<id>`, `x.com/home`) no longer parse
  as handles called `i` or `home`. xAI's own citations use the handle-less form, so a
  model passing one back as a filter silently scoped the search to an account that does
  not exist. `_source_name` already knew this; the parser now does too.
- An over-long handle in URL form is rejected instead of truncated to its first 15
  characters, which produced a wrong but entirely plausible handle. The bare form
  already rejected it; the two forms now agree.
- A handle filter that parses to nothing no longer searches all of X. An allow-list
  failing open is a scope violation, so it is refused whatever its source, as is a
  call-level exclusion. Only the instance-wide exclusion default degrades to a note, so
  one bad valve cannot brick every search on a server.
- `allowed_handles` + `excluded_handles` in one call is now rejected before parsing, so
  an unparseable exclusion can no longer swallow the conflict and leave the search
  running with half of what was asked for.
- Host matching is case-insensitive throughout: `https://X.com/elonmusk` labels as
  `@elonmusk` rather than `X.com`.
- A 200 response whose body is valid JSON but not an object no longer crashes the tool
  with an unhandled `AttributeError`; it is reported like any other unreadable body.

## 1.0.0

First working release.

- `search_x` wraps xAI's server-side `x_search` tool on the Responses API, giving any
  tool-calling model in Open WebUI live access to X posts.
- Supports every parameter xAI documents for `x_search`: `allowed_x_handles`,
  `excluded_x_handles`, `from_date`, `to_date`, `enable_image_understanding`,
  `enable_video_understanding`.
- Results preserve the search model's own citation numbering, and separate sources it
  cited from sources it merely consulted.
- Emits Open WebUI status events while searching and citation events for each source.
- Per-message call gate: refuses fan-out siblings issued in one batch, allows genuine
  retries after a result has been read, and caps runaway loops.
- Configuration via valves, including model, date/handle defaults, media understanding,
  timeouts, and both gate thresholds.

### Notes on the shape of this release

`search_x_recent` existed during development and was merged into `search_x(days=...)`.
Two overlapping tools led models to call both for a single question to get "a recent
view and a background view" — one investigation's worth of material for two
investigations' worth of cost and latency. Collapsing them to one tool removed the
affordance, which worked where instructions in the tool description had not.
