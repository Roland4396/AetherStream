# Hydrogen shared-Gemini quota cycles

Hydrogen (4h) is covered by the upstream `gemini-5h` and `gemini-weekly`
allowances, shared with ordinary Gemini models. It does **not** have an
independent five-hour budget. Argon (4a) keeps its existing independent
`gemini-argon-wtu-weekly` schedule. No Argon or Claude state is reinitialized.

## Opt in

Add `"hydrogen": {"enabled": true, "model": "gemini-4-hydrogen-low"}` inside
`quota_keeper` in the hot-reloaded runtime flags. Missing configuration is off.
Only `enabled` and `model` are accepted; low/medium/high/xhigh are allowed.
The global keeper switch and minimum 60-second local schedule-check interval
still apply. This is not one upstream request per minute.

The existing singleton, lifecycle lease and state lock own all three schedules.
No new timer, thread, background worker, public port or inference route exists.
`hydrogen` is an additive nested account state, retaining schema version 1.
Old versions retain that state across rollback; resuming does not erase guards.

## Decision and generation gates

* Read fresh successful native quota diagnostics and enabled model catalogue.
  Require both exact shared-Gemini buckets. Missing, ambiguous, malformed,
  stale or disabled quota never authorizes generation. Do not substitute
  Argon's week, third-party quota, a cached percentage or an inferred budget.
* Weekly exhaustion wins even if the short window says 100% remaining. Wait
  for the weekly reset; there is no paid-credit overage or alternate-account
  fallback. Existing usage by **any** model sharing Gemini quota suppresses
  the keeper's wakeup until the reported short/weekly boundary.
* A reset fixed across separate observations proves a running cycle even
  when remaining quota is 100%. A deadline moving with `now + 5h` or
  `now + 7d` is an unused-window placeholder, not a fixed reset date.
* Only an available, full, unused short window with positive weekly quota can
  start one tiny real quota-consuming request. It uses the selected exact
  account/provider and configured Hydrogen model through GProxy's native
  model test: `ping`, output limit 16, one core attempt. No direct Google
  client, CLI wrapper, Stream text transforms, retries or account rotation.
* Persist the request guard and uncertainty before that POST. HTTP success or
  a balance decrease alone is not cycle confirmation. Verify the fixed clock
  using metadata; ambiguous delivery survives expiry, restart and resume and
  cannot authorize another completion. Authentication/permission errors pause
  the new family immediately. Other transport errors get bounded metadata-only
  confirmation, then pause rather than retry generation.
* A pre-existing 401/403 generation result in Claude, Argon or Hydrogen blocks
  initialization/generation of Hydrogen on that account. Quota metadata can
  remain readable and report 100% on a service-disabled account. A historical
  non-permission Claude pause does not erase or merge independent schedules.

Independent user requests can race with the final quota read; this mechanism
prevents duplicate **keeper** work, not unrelated concurrent user requests.
Official service permissions and limits remain authoritative.

## Private administration

The existing authenticated `GET /admin/quota-keeper` adds global Hydrogen
configuration and per-account safe status. Reading it performs no upstream
requests. The authenticated
`POST /admin/quota-keeper/credentials/{id}/hydrogen/resume` schedules a metadata
recheck only, preserving paid-attempt guards and uncertainty and leaving
Claude/Argon untouched. Existing permission-refusal evidence still prevents
an automatic restart; investigate authorization rather than clearing guards.

## Release and validation

Build an immutable image and deploy with the rolling-release controller. Add
runtime configuration only after the compatible version is serving, using the
shared runtime-settings lock, a revision check and inode-preserving write.
Never replace an active application container or restart stable ingress.

Tests use synthetic quota and model-test responses, no paid inference or GPU.
They cover shared-budget ownership, weekly exhaustion, fixed versus moving
clocks, single dispatch, uncertain delivery across restart/resume, permissions,
account deduplication, lifecycle shutdown, cross-process locks, state rollback
and private status/resume endpoints. Run the complete offline suite and real
local Nginx SSE/WebSocket promotion/rollback before production promotion.
