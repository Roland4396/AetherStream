"""Stream-owned feature policies. Pool alone selects channels and prefixes.

This module consumes a Pool selection descriptor; it never chooses a channel,
rewrites a public model, reads an account credential, or calls a provider.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import re
from typing import Any

from aetherstream.routing import request_mode

class RegistryError(ValueError):
    pass

POLICY_KEYS = frozenset({"streaming", "early_stop", "injections", "filters", "request",
                        "runtime", "session", "latest_wins", "adapter", "terminal_tool"})

def merge(base: dict, override: dict) -> dict:
    """Objects merge; scalars and arrays replace (including false/empty)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_fields(payload: dict, rule: dict) -> None:
    """Generic dotted-field changes applied after selecting the actual target."""
    for path in rule.get("drop", []):
        parts = path.split('.')
        value = payload
        parents = []
        for part in parts[:-1]:
            if not isinstance(value.get(part), dict):
                value = None
                break
            parents.append((value, part))
            value = value[part]
        if value is not None:
            value.pop(parts[-1], None)
            for parent, key in reversed(parents):
                if not parent[key]:
                    parent.pop(key)
    for path, new_value in rule.get("set", {}).items():
        parts = path.split('.')
        value = payload
        for part in parts[:-1]:
            if not isinstance(value.get(part), dict):
                value[part] = {}
            value = value[part]
        value[parts[-1]] = copy.deepcopy(new_value)


def matches(model: str, selector: dict) -> bool:
    if not selector:
        return False
    # Every specified condition must match. An explicit 'any' is OR.
    if "any" in selector and not any(matches(model, s) for s in selector["any"]):
        return False
    if "exact" in selector and model not in selector["exact"]:
        return False
    if "prefix" in selector and not model.startswith(selector["prefix"]):
        return False
    if "regex" in selector and re.search(selector["regex"], model, re.IGNORECASE) is None:
        return False
    return True


def validate_selector(selector: Any) -> None:
    if not isinstance(selector, dict) or set(selector) - {"any", "exact", "prefix", "regex"}:
        raise RegistryError("invalid model selector")
    if "regex" in selector:
        re.compile(selector["regex"], re.IGNORECASE)
    if "prefix" in selector and not isinstance(selector["prefix"], str):
        raise RegistryError("model prefix must be a string")
    if "exact" in selector and (not isinstance(selector["exact"], list)
                                or not all(isinstance(x, str) and x for x in selector["exact"])):
        raise RegistryError("exact models must be a list of names")
    if "any" in selector:
        if not isinstance(selector["any"], list) or not selector["any"]:
            raise RegistryError("any selector must be a nonempty list")
        for child in selector["any"]:
            validate_selector(child)


def validate_policy(policy: Any) -> None:
    if not isinstance(policy, dict) or set(policy) - POLICY_KEYS:
        raise RegistryError("unknown channel policy field")
    # Existing schema-1 readers ignore new request-runtime feature settings,
    # so config and shared state both remain usable during rollback.
    runtime = policy.get('runtime', {})
    if isinstance(runtime, dict) and 'output_tool' in runtime:
        raise RegistryError('Configure output_tool at the runtime root, not inside channel policies')
    guard = runtime.get('normal_completion_guard', {}) if isinstance(runtime, dict) else {}
    if not isinstance(guard, dict) or set(guard) - {'enabled', 'scope', 'threshold', 'account_ids'}:
        raise RegistryError('invalid normal_completion_guard policy')
    if 'enabled' in guard and not isinstance(guard['enabled'], bool):
        raise RegistryError('normal_completion_guard enabled must be boolean')
    if 'threshold' in guard and (type(guard['threshold']) is not int or not 1 <= guard['threshold'] <= 100):
        raise RegistryError('normal_completion_guard threshold must be an integer from 1 to 100')
    if 'scope' in guard and (not isinstance(guard['scope'], str) or not 0 < len(guard['scope']) <= 128):
        raise RegistryError('normal_completion_guard scope must be nonempty')
    if 'account_ids' in guard and (not isinstance(guard['account_ids'], list) or not guard['account_ids']
                                 or not all(isinstance(x, str) and 0 < len(x) <= 128 for x in guard['account_ids'])):
        raise RegistryError('normal_completion_guard account_ids must be nonempty names')
    if guard.get('enabled') and not all(k in guard for k in ('scope', 'threshold', 'account_ids')):
        raise RegistryError('enabled normal_completion_guard requires scope, threshold and account_ids')
    stop = policy.get("early_stop", {})
    if not isinstance(stop, dict) or set(stop) - {"enabled", "action", "tags", "case_sensitive", "content_guard"}:
        raise RegistryError("invalid early_stop policy")
    if "action" in stop and stop["action"] != "disconnect":
        raise RegistryError("early_stop action must be disconnect")
    for key in ("enabled", "case_sensitive", "content_guard"):
        if key in stop and not isinstance(stop[key], bool):
            raise RegistryError("early_stop switches must be booleans")
    if "tags" in stop and (not isinstance(stop["tags"], list) or not all(
            isinstance(tag, str) and 0 < len(tag) <= 256 for tag in stop["tags"])):
        raise RegistryError("invalid early_stop tags")
    if len(json.dumps(stop)) > 4096:
        raise RegistryError("early_stop policy is too large")
    streaming = policy.get("streaming", {})
    if not isinstance(streaming, dict) or set(streaming) - {
            "nonstream_to_stream", "force_upstream", "json_keepalive", "deadline_seconds", "exact_dedupe"}:
        raise RegistryError("invalid streaming policy")
    if streaming.get("force_upstream", "auto") not in {"auto", "stream", "json"}:
        raise RegistryError("invalid upstream stream mode")
    for key in ("nonstream_to_stream", "json_keepalive", "exact_dedupe"):
        if key in streaming and not isinstance(streaming[key], bool):
            raise RegistryError("streaming switches must be booleans")
    if "deadline_seconds" in streaming and (
            isinstance(streaming["deadline_seconds"], bool)
            or not isinstance(streaming["deadline_seconds"], (int, float))
            or not 0 <= streaming["deadline_seconds"] <= 3600):
        raise RegistryError("invalid stream deadline")
    injections = policy.get("injections", {})
    if not isinstance(injections, dict) or not all(isinstance(x, str) and isinstance(v, bool)
                                                 for x, v in injections.items()):
        raise RegistryError("injections must be named boolean switches")



def validate_config(cfg, globals_):
    if not isinstance(cfg, dict) or cfg.get('schema') != 1:
        raise RegistryError('channel_policies schema=1 is required')
    if set(cfg) - {'schema','defaults','profiles','model_rules','channels','account_overrides','dedupe_credentials'}:
        raise RegistryError('unknown channel_policies field')
    validate_policy(merge(cfg.get('defaults', {}), globals_))
    profiles = cfg.get('profiles', {})
    for value in profiles.values():
        validate_policy(value)
    for rule in cfg.get('model_rules', []):
        validate_selector(rule.get('match'))
        validate_policy(rule.get('policy', {}))
    for cid, channel in cfg.get('channels', {}).items():
        if not cid or not isinstance(channel, dict) or set(channel) - {'profiles','policy','routes'}:
            raise RegistryError('Stream policy must not contain channel selection fields')
        validate_policy(channel.get('policy', {}))
        for profile in channel.get('profiles', []):
            if profile not in profiles:raise RegistryError('unknown policy profile')
        for rid, route in channel.get('routes', {}).items():
            if not rid or not isinstance(route, dict) or set(route) - {'profiles','policy','dedupe'}:
                raise RegistryError('Stream policy must not contain route selection fields')
            validate_policy(route.get('policy', {}))
            for profile in route.get('profiles', []):
                if profile not in profiles:raise RegistryError('unknown policy profile')
    for value in cfg.get('account_overrides', {}).values():
        if not isinstance(value, dict) or set(value) - {'early_stop'}:
            raise RegistryError('post-selection overrides only support early_stop')
        validate_policy(value)
    try:
        request_mode.validate_locations(cfg, globals_)
    except request_mode.ModeConfigError as exc:
        raise RegistryError(str(exc)) from exc


def global_policy(lookup):
    # An old slot must not hash an unknown format-only feature into JSON locks.
    # Keep its config outside the legacy policy projection; attach a selected
    # request snapshot below, and remove that snapshot from transport identities.
    output_tool = lookup('output_tool')
    if output_tool is not None:
        from aetherstream.features.output_tool import validate_settings
        try:
            validate_settings(output_tool)
        except (ValueError, TypeError, AttributeError) as exc:
            raise RegistryError('invalid output_tool configuration: ' + str(exc)) from exc
    value = {k: copy.deepcopy(lookup(k)) for k in ('early_stop','streaming','filters')
             if lookup(k) is not None}
    value['runtime'] = {k: copy.deepcopy(lookup(k)) for k in ('claude','injections')
                        if lookup(k) is not None}
    return value


def enrich_selection(lookup, selection):
    cfg = copy.deepcopy(lookup('channel_policies'))
    if cfg is None:
        return selection  # Explicit legacy rollout compatibility, not a selector.
    globals_ = global_policy(lookup)
    validate_config(cfg, globals_)
    cid, rid = selection['channel'], selection['route']
    # Pool owns valid selections. Missing feature overrides mean defaults,
    # not a second channel/model admission list maintained in Stream.
    channel = cfg.get('channels', {}).get(cid, {})
    route = channel.get('routes', {}).get(rid, {})
    upstream = selection.get('upstream_model')
    if not isinstance(upstream, str) or not upstream:
        raise RegistryError('Pool selection is missing upstream_model')
    policy = merge(cfg.get('defaults', {}), globals_)
    for profile in channel.get('profiles', []):policy = merge(policy, cfg['profiles'][profile])
    policy = merge(policy, channel.get('policy', {}))
    for profile in route.get('profiles', []):policy = merge(policy, cfg['profiles'][profile])
    policy = merge(policy, route.get('policy', {}))
    for rule in cfg.get('model_rules', []):
        if rule.get('transports') and selection['transport'] not in rule['transports']:continue
        if rule.get('channels') and cid not in rule['channels']:continue
        if matches(upstream, rule['match']):policy = merge(policy, rule.get('policy', {}))
    mode = request_mode.resolve(cfg, cid, upstream)
    policy.setdefault('streaming', {})['force_upstream'] = mode['mode']
    policy['streaming'].pop('nonstream_to_stream', None)
    from aetherstream.features.output_tool import selected_settings
    output_tool = selected_settings(lookup('output_tool'), upstream, selection['transport'])
    if output_tool is not None:
        policy.setdefault('runtime', {})['output_tool'] = output_tool
    if policy.get('session', {}).get('model') == '$upstream':policy['session']['model'] = upstream
    policy['account_overrides'] = copy.deepcopy(cfg.get('account_overrides', {}))
    policy['_local_policy'] = True
    dedupe = copy.deepcopy(route.get('dedupe', {}))
    if dedupe.get('model') == '$upstream':dedupe['model'] = upstream
    elif dedupe.get('model') == '$public':dedupe['model'] = selection['model']
    if dedupe.get('gemini_payload') and 'gemini' in upstream.lower():dedupe.update(kind='payload', shape='chat')
    revision_input = {'config': cfg, 'globals': globals_}
    if output_tool is not None:
        revision_input['output_tool'] = output_tool
    revision = hashlib.sha256(json.dumps(revision_input, sort_keys=True,
                                         ensure_ascii=False,separators=(',', ':')).encode()).hexdigest()[:20]
    return {**selection, 'policy':policy, 'policy_revision':revision, 'request_mode':mode, '_local_dedupe':True,
            'dedupe':dedupe, 'dedupe_credentials':copy.deepcopy(cfg.get('dedupe_credentials', {}))}


def dedupe_identity(plan, payload, headers):
    cfg = plan.get('dedupe', {})
    if not cfg:
        identity = {'scope':'stream-channel-json-v1','route':plan['route'],
                    'upstream_model':plan['upstream_model'],
                    'policy':request_mode.compatible_dedupe_policy(plan['policy']),'request':payload}
        shape = 'chat'
    else:
        body = copy.deepcopy(payload);body['model'] = cfg.get('model', plan['model']);shape = cfg['shape']
        if cfg.get('kind') == 'payload':identity = body
        else:
            original_headers = {k:v for k,v in headers.items() if not k.lower().startswith('x-acc-')}
            auth_name = cfg.get('auth_header','Authorization')
            for key in list(original_headers):
                if key.lower() in {'authorization','x-api-key'}:original_headers.pop(key)
            credential = plan.get('dedupe_credentials', {}).get(cfg.get('credential'), '')
            if isinstance(credential,dict):credential = os.environ.get(credential.get('env',''), '')
            if cfg.get('caller_authorization'):
                credential = next((v for k,v in headers.items() if k.lower() == 'x-acc-passthrough-authorization'), '')
                prefix = ''
            else:prefix = cfg.get('auth_prefix','Bearer ')
            if credential:original_headers[auth_name] = prefix + credential
            original_headers.update(cfg.get('headers', {}))
            identity = {'scope':cfg['scope'],'url':cfg['url'],'headers':original_headers,'request':body}
    digest = hashlib.sha256(json.dumps(identity,ensure_ascii=False,sort_keys=True,
                                       separators=(',', ':')).encode('utf-8',errors='replace')).hexdigest()
    return {'key':digest,'shape':shape,'revision':plan['revision']}
