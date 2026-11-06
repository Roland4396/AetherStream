# Independent Argon weekly keepalive

The existing quota-keeper singleton can maintain the Argon weekly cycle independently of the Claude five-hour schedule. No additional timer, background task, public inference route or direct Google client is introduced.

## Opt-in configuration

```json
{
  "quota_keeper": {
    "enabled": true,
    "check_interval_sec": 60,
    "argon_weekly": {
      "enabled": true,
      "model": "gemini-4-argon-low"
    }
  }
}
```

Absent `argon_weekly` defaults to disabled. Only `enabled` (boolean) and `model` (Argon low, medium or high) are accepted. Invalid settings disable this feature without disabling the independently configured Claude schedule. There is no fallback to another variant, account or ordinary Gemini quota. The global keeper switch also applies.

## Quota and paid-request policy

* Fresh native GProxy quota diagnostics must show an enabled configured model and the separate `gemini-argon-wtu-weekly` bucket. Legacy normalized quota omits that bucket and must not be used as a substitute.
* A non-full balance, including exhaustion, suppresses another generation: wait until that account's reported reset plus the existing grace period. Normal use therefore suppresses the keepalive. A full balance with a moving seven-day deadline does not by itself prove an active cycle.
* An eligible full unused balance allows one tiny **real, quota-consuming** request through GProxy's exact-account `/admin/api/models/test`. The currently deployed GProxy implementation sends `ping`, native `maxOutputTokens=16`, with one attempt, and its native session/TLS identity. Stream ACT/Kunlun transforms are not involved. Maintain these invariants when upgrading GProxy.
* Persist the account's request guard and delivery uncertainty before that POST. A later fresh quota reading must confirm cycle usage; HTTP success alone is insufficient. A fixed reset clock across distinct observations confirms the cycle even if the displayed balance remains full; a balance decrease alone is not clock confirmation. No retries of an uncertain paid request, including after a restart, elapsed week or explicit resume.
* Failed quota reads back off and eventually pause only the Argon schedule. Accounts without Argon permission receive a daily metadata recheck, never inference. Delivery failures receive a quota-only verification; unresolved delivery pauses.
* Duplicate imports of the same email share one state and one paid guard. The same kernel state lock and lifecycle singleton lease protect Claude and Argon across processes and releases. Stop waits for an in-flight request; durable uncertainty survives forced timeout or crash.

A user request racing immediately after the fresh quota read cannot be atomically excluded across independent clients; this mechanism prevents keeper duplicates, not unrelated concurrent user requests. Diagnostic and generation requests themselves are never automatically replayed by this client.

## State, observability and recovery

Argon state is additive under each existing account's `argon_weekly`; the schema stays at version 1. Existing Claude fields, guards, pauses and resets are preserved. Old readers retain the nested state, enabling rollback without deleting schedules.

The existing private `GET /admin/quota-keeper` exposes safe metrics only, including top-level Argon configuration and per-account weekly status. It performs no network calls. Replies, credentials, account labels and raw diagnostic/model-experiment data are not stored in the new state or logs.

`POST /admin/quota-keeper/credentials/{id}/argon/resume`, protected by the existing separate admin token, schedules a quota recheck only. It retains paid-attempt guards and uncertainty, and does not resume or modify Claude. If delivery remains unconfirmed and quota remains full, rechecking pauses again rather than issuing a completion. Investigate metadata or delivery evidence before any manual state repair; do not erase guards to force a retry.

Deploy the compatible code first with the new option absent/disabled using the rolling-release controller, then enable the option by a locked, revision-checked, inode-preserving runtime configuration update. No GProxy, pool, Tavern or stable ingress restart is required.
