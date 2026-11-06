# Claude quota keeper: native model and cycle verification

## Fix scope

The retired implementation directly requested the fixed `claude-opus-4-6-thinking` name and marked any HTTP-successful response with nonempty candidate text as successful, even without a response model or token usage. It could not prove that the requested model actually generated. Those legacy results remain historical records, not verified model identity. The private status API now projects empty-usage results as unverified.

Claude keepalive now uses GProxy's existing native exact-provider/exact-credential management generation, preserving its client/session identity and native model translation. It never reveals an account secret, calls Google directly, passes through Stream's story transforms, or silently falls back to another model/account. GProxy currently sends `ping` with a native output cap of 16 tokens and one core attempt.

## Configuration

```json
{
  "quota_keeper": {
    "enabled": true,
    "check_interval_sec": 60,
    "claude": {
      "enabled": true,
      "model": "claude-opus-5-5-low"
    }
  }
}
```

The selected model is read from configuration, with no hardcoded per-request model override. The default is the current Opus 5.5 low alias. A fresh native catalogue must explicitly offer the exact configured model before a completion may be submitted. Missing, disabled, malformed or stale model/quota evidence cannot authorize spending. Model withdrawal pauses that account without generating. Updating the configured model never authorizes account/model fallback or clears an existing paid guard.

`claude.enabled=false` independently disables Claude work; Argon retains its own setting. The global keeper switch disables both. Invalid Claude settings fail closed only for Claude. Existing public chat routing and prompts are unaffected.

## Two separate proofs

1. **Generation evidence:** native management reports success with a valid nonzero output/reasoning token metric. HTTP success, a requested-model echo, nonempty arbitrary reply text, or input tokens alone are insufficient. Reply text and arbitrary errors are not logged or persisted. The response DTO's `model` is the request-name echo and is never treated as an actual response model.
2. **Cycle confirmation:** the relevant reset timestamp remains fixed across two distinct fresh observations separated by at least 15 seconds, allowing a two-second rounding tolerance. Claude confirms its five-hour clock; Argon confirms its independent weekly clock. Displayed usage can be zero or rounded away. Usage can suppress an unnecessary paid request but does not itself confirm a running clock.

An unused quota endpoint can return `observation time + five hours/seven days` on every query. That moving placeholder is not an active cycle. Cached snapshots are compared using their own observation timestamp; merely aging a cached unused deadline cannot turn it into running-cycle evidence. A shortened but unconfirmed full-balance clock is observed again without sending inference. A pre-request quota sample is retained as a baseline for post-request cycle comparison.

Persist paid-delivery uncertainty before POST. After a response, perform quota-only verification. A response without usage remains `response_unverified`; ambiguous delivery retains the guard. A nonzero balance usage with a still-sliding clock is not falsely reported as successful: bounded metadata-only clock checks eventually pause. Network errors use bounded metadata backoff, never inference replay.

Uncertainty survives a crash, restart, guard expiry and explicit resume. Successful cycle confirmation clears uncertainty and schedules the account's real reset plus grace. An expired guard alone is never proof that a paid attempt did not arrive.

## State, recovery and deployment

Schema version remains 1, with additive fields and no schedule resets. Existing paused accounts are not silently resumed en masse. `/admin/quota-keeper` is disk-only and separately authenticated. `POST /admin/quota-keeper/credentials/{id}/resume` schedules a recheck without clearing guards; it cannot force a duplicate paid request. A freshly proven running cycle can resolve an old unconfirmed pause after an explicit resume.

Use an immutable-image rolling release. The old reader preserves additive fields, and rollback does not erase guards. Test with network-disabled mocks plus local real Nginx SSE/WebSocket promotion, rollback and timeout probes. Do not use paid inference as a regression test. An explicitly authorized one-account diagnostic generation is recorded separately from automated test results.
