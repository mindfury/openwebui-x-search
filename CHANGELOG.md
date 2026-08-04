# Changelog

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
