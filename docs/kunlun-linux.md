# Kunlun Linux request envelope

This optional Stream feature is selected by exact channel, public model and chat
transport. It does not install or change a Tavern preset, plugin, settings file,
worldbook or chat. The request transformer remains scoped to a request-owned copy.

## Scene insertion default

The supplied Linux 1.5 preset supplies an insertion count of three through its
scene-configuration prompt. When using only its script in Stream, configure that
missing value explicitly:

```json
{
  "kunlun_linux": {
    "enabled": true,
    "default_insert_count": 3,
    "match": {
      "channels": ["free"],
      "public_models": ["free/gemini-3.8-flash-high"]
    },
    "transports": ["chat"]
  }
}
```

The default applies only when the request does not provide an insertion count.
An explicit count in the request, including zero, takes precedence. The original
gate requiring at least two historical user inputs is unchanged. Corpus contents,
selection, ordering, character-roster extraction and message framing are unchanged.
This option supplies the count only; it does not import the preset's other prompts.

Omitting the option retains the original script's no-configuration behavior.
Diagnostics report the count source, historical-user count, inserted count and
skip reason without including story or chat content. Configured defaults enter
the exact-request identity, while random tool names do not.

## Rollout

Deploy code that accepts this optional field before adding it to the shared
runtime-flags file. The preceding release rejects unknown feature fields, so add
the new field only after that release has drained. To roll back to code predating
the field, first remove it under the runtime-flags lock.
