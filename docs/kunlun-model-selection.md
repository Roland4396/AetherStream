# Configuration-owned Kunlun model-family selection

`kunlun_linux` still owns the existing preset, enabled flag and supported
transport. Its legacy exact-name matching remains unchanged. Do not change the
preset or copy prompt text to add another model.

An optional runtime-root setting extends selection to whole model families:

```json
{
  "kunlun_linux_model_rules": [
    {"match": {"regex": "gemini-4-(?:argon|carbon)(?:$|[-_/])"}}
  ]
}
```

Rules use the existing Stream selector syntax (`exact`, `prefix`, `regex`,
`any`). A rule matches the public model or the Pool-resolved upstream model;
regular expressions are case-insensitive. Optional `channels` restricts a rule
to the listed Pool channels. Without `channels`, new channel aliases also work.
Rules cannot select accounts, rewrite models or change upstream routes.

The example covers Argon and Carbon names, decorated prefixes and future tiers,
without matching names such as `gemini-4-carbonated`. Existing exact-name
exceptions, including the configured free Flash model, continue working.
The enabled flag and chat-only transport gate still apply. Kunlun and the
generic output-tool envelope remain mutually exclusive, with Kunlun first.
Unrelated feature policies, request modes and dedupe identities are unchanged.

The new root key is intentionally outside the legacy preset object: older
readers reject unknown fields inside that object but safely ignore unknown
runtime-root features. New readers accept old configuration. On rollback, an
older reader retains the legacy exact-name targets and ignores the additional
family rules. No shared-state migration, background job, inference retry or
provider request is introduced.

Deploy code through the rolling release controller before adding the optional
root setting. Edit the shared flags with its existing revision/file-lock and
inode-preserving write contract; do not replace the bind-mounted file.
