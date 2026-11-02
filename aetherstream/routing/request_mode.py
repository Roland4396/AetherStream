"""One request-mode setting, owned by a channel or an exact model in that channel.

Other feature policies still support profiles. Request mode deliberately does
not: it has only channel -> model precedence, and otherwise follows the caller.
The existing force_upstream wire values keep old release slots compatible.
"""
from __future__ import annotations

import copy


MODES = frozenset({'auto', 'stream', 'json'})
LABELS = {'auto': '跟随酒馆', 'stream': '固定流式', 'json': '固定非流式'}


class ModeConfigError(ValueError):
    pass


def _streaming(policy):
    return policy.get('streaming', {}) if isinstance(policy, dict) else {}


def validate_locations(config: dict, globals_: dict) -> None:
    """Refuse hidden conversions instead of silently accepting ignored switches.

    Inert auto/false values are accepted only for the expand/contract rollout;
    they cannot enable any conversion and are removed by the migration.
    """
    def no_mode(policy, location):
        settings = _streaming(policy)
        if settings.get('force_upstream', 'auto') != 'auto':
            raise ModeConfigError(f'{location}: request mode must be set on a channel or exact model')
        if settings.get('nonstream_to_stream', False) is not False:
            raise ModeConfigError(f'{location}: nonstream_to_stream was replaced by one request-mode selector')

    no_mode(globals_, 'global streaming')
    no_mode(config.get('defaults', {}), 'policy defaults')
    for name, profile in config.get('profiles', {}).items():
        no_mode(profile, f'profile {name}')
    for cid, channel in config.get('channels', {}).items():
        if _streaming(channel.get('policy', {})).get('nonstream_to_stream', False) is not False:
            raise ModeConfigError(f'channel {cid}: use force_upstream, not a second conversion switch')
        for rid, route in channel.get('routes', {}).items():
            no_mode(route.get('policy', {}), f'route {cid}/{rid}')
    seen = set()
    for rule in config.get('model_rules', []):
        settings = _streaming(rule.get('policy', {}))
        if settings.get('nonstream_to_stream', False) is not False:
            raise ModeConfigError('model rule: use force_upstream, not a second conversion switch')
        if 'force_upstream' not in settings:
            continue
        channels, selector = rule.get('channels'), rule.get('match', {})
        if (not isinstance(channels, list) or len(channels) != 1
                or not isinstance(channels[0], str) or not channels[0]
                or set(selector) != {'exact'} or not selector['exact']
                or rule.get('transports')):
            raise ModeConfigError('request-mode model rules require one channel and exact upstream model names')
        for model in selector['exact']:
            key = (channels[0], model)
            if key in seen:
                raise ModeConfigError('duplicate request-mode rule for the same channel and model')
            seen.add(key)


def resolve(config: dict, channel: str, upstream_model: str) -> dict:
    settings = _streaming(config.get('channels', {}).get(channel, {}).get('policy', {}))
    mode = settings.get('force_upstream', 'auto')
    source = 'channel' if 'force_upstream' in settings else 'default'
    for rule in config.get('model_rules', []):
        setting = _streaming(rule.get('policy', {}))
        if ('force_upstream' in setting and rule.get('channels') == [channel]
                and upstream_model in rule.get('match', {}).get('exact', [])):
            mode, source = setting['force_upstream'], 'model'
    if mode not in MODES:
        raise ModeConfigError('invalid request mode')
    return {'mode': mode, 'label': LABELS[mode], 'source': source,
            'channel': channel, 'upstream_model': upstream_model,
            'nonstream_request_uses_stream': mode == 'stream',
            'stream_request_uses_stream': mode != 'json'}


def configured_models(config: dict, channel: str) -> list[str]:
    return sorted({model for rule in config.get('model_rules', [])
                   if rule.get('channels') == [channel]
                   and 'force_upstream' in _streaming(rule.get('policy', {}))
                   for model in rule.get('match', {}).get('exact', [])})


def set_mode(flags: dict, *, channel: str, model: str | None, mode: str) -> dict:
    """Change exactly one setting; 'inherit' removes only that model override."""
    if (not isinstance(channel, str) or not channel.strip() or len(channel) > 128
            or any(ord(c) < 32 for c in channel)):
        raise ModeConfigError('channel must be a nonempty ID of at most 128 characters')
    if model is not None and (not isinstance(model, str) or not model.strip() or len(model) > 512
                              or any(ord(c) < 32 for c in model)):
        raise ModeConfigError('model must be an exact upstream name of at most 512 characters')
    if mode not in MODES and not (model is not None and mode == 'inherit'):
        raise ModeConfigError('choose auto, stream, or json; model overrides may also inherit')
    result = copy.deepcopy(flags)
    config = result['channel_policies']
    if model is None:
        policy = config.setdefault('channels', {}).setdefault(channel, {}).setdefault('policy', {})
        policy.setdefault('streaming', {})['force_upstream'] = mode
        return result
    rules = config.setdefault('model_rules', [])
    for rule in list(rules):
        settings = _streaming(rule.get('policy', {}))
        if (rule.get('channels') != [channel] or model not in rule.get('match', {}).get('exact', [])
                or 'force_upstream' not in settings):
            continue
        # Splitting a multi-model rule preserves every other model/feature.
        if len(rule['match']['exact']) > 1:
            other = copy.deepcopy(rule)
            other['match']['exact'].remove(model)
            rules.insert(rules.index(rule), other)
            rule['match']['exact'] = [model]
        settings.pop('force_upstream')
        if not settings:
            rule['policy'].pop('streaming', None)
        if not rule['policy']:
            rules.remove(rule)
        if mode != 'inherit':
            if rule in rules:
                rule['policy'].setdefault('streaming', {})['force_upstream'] = mode
                return result
            break
    if mode != 'inherit':
        rules.append({'match': {'exact': [model]}, 'channels': [channel],
                      'policy': {'streaming': {'force_upstream': mode}}})
    return result


def compatible_dedupe_policy(policy: dict) -> dict:
    """A hash-only legacy projection, never a second request-mode control.

    Old slots hashed an inert nonstream_to_stream=false field. Retaining that
    spelling solely in the identity projection prevents duplicate paid work
    across the rolling migration. Execution never reads this projection.
    """
    result = copy.deepcopy(policy)
    result.setdefault('streaming', {})['nonstream_to_stream'] = False
    result['streaming'].setdefault('force_upstream', 'auto')
    claude = result.get('runtime', {}).get('claude')
    if isinstance(claude, dict) and isinstance(claude.get('stream'), dict):
        claude['stream'].setdefault('nonstream_to_stream', False)
    return result


def migrate_flags(flags: dict, *, keep_inert_legacy: bool = False) -> tuple[dict, dict]:
    """Lift existing profile/route modes to explicit channel settings.

    Never guess when routes of one channel disagree, or when an active global
    conversion needs a human decision. The deployment config is staged first;
    this function does not write files or touch a running service.
    """
    result = copy.deepcopy(flags)
    config = result['channel_policies']
    profiles = config.get('profiles', {})
    report = {'channel_modes': {}, 'removed_mode_fields': []}

    def mode_for(policies):
        mode = 'auto'
        for policy in policies:
            settings = _streaming(policy)
            if settings.get('nonstream_to_stream', False) is not False:
                raise ModeConfigError('disable the old conversion boolean before migrating; no implicit migration')
            mode = settings.get('force_upstream', mode)
        return mode

    # Global conversion is not expanded into a hidden per-channel default.
    for policy in (result, config.get('defaults', {})):
        if mode_for([policy]) != 'auto':
            raise ModeConfigError('global forced modes must first be assigned to explicit channels')
    channel_modes = {}
    for cid, channel in config.get('channels', {}).items():
        chain = [result, config.get('defaults', {})]
        chain += [profiles[name] for name in channel.get('profiles', [])]
        chain += [channel.get('policy', {})]
        routes = channel.get('routes', {})
        modes = set()
        for route in routes.values():
            policies = chain + [profiles[name] for name in route.get('profiles', [])] + [route.get('policy', {})]
            modes.add(mode_for(policies))
        if not modes:
            modes.add(mode_for(chain))
        if len(modes) != 1:
            raise ModeConfigError(f'channel {cid} has conflicting route modes; assign exact model settings first')
        channel_modes[cid] = modes.pop()
    for rule in config.get('model_rules', []):
        settings = _streaming(rule.get('policy', {}))
        if settings.get('nonstream_to_stream', False) is not False:
            raise ModeConfigError('model conversion booleans must first be replaced with explicit modes')
        if settings.get('force_upstream', 'auto') != 'auto':
            if len(rule.get('channels', [])) != 1 or set(rule.get('match', {})) != {'exact'}:
                raise ModeConfigError('scope a forced model mode to one channel and exact names before migrating')

    def strip(policy, location, *, keep_force=False):
        settings = _streaming(policy)
        for key in ('nonstream_to_stream', 'force_upstream'):
            if key in settings and not (keep_force and key == 'force_upstream'):
                settings.pop(key)
                report['removed_mode_fields'].append(location + '.streaming.' + key)
        if not settings and isinstance(policy, dict):
            policy.pop('streaming', None)

    strip(result, 'global')
    strip(config.get('defaults', {}), 'defaults')
    for name, profile in profiles.items():
        strip(profile, 'profiles.' + name)
    for cid, channel in config.get('channels', {}).items():
        strip(channel.get('policy', {}), 'channels.' + cid, keep_force=True)
        for rid, route in channel.get('routes', {}).items():
            strip(route.get('policy', {}), f'channels.{cid}.routes.{rid}')
        if channel_modes[cid] != 'auto':
            channel.setdefault('policy', {}).setdefault('streaming', {})['force_upstream'] = channel_modes[cid]
            report['channel_modes'][cid] = channel_modes[cid]
    for i, rule in enumerate(config.get('model_rules', [])):
        scoped = len(rule.get('channels', [])) == 1 and set(rule.get('match', {})) == {'exact'}
        strip(rule.get('policy', {}), f'model_rules.{i}', keep_force=scoped)
    legacy = result.get('claude', {}).get('stream', {})
    if 'nonstream_to_stream' in legacy:
        if legacy['nonstream_to_stream'] is not False:
            raise ModeConfigError('disable the legacy Claude global conversion before migrating')
        legacy.pop('nonstream_to_stream')
        report['removed_mode_fields'].append('claude.stream.nonstream_to_stream')
    if keep_inert_legacy:
        # Old slots keep stable shared-request identities while they drain.
        # The new reader accepts only these non-converting inert values.
        result.setdefault('streaming', {}).update(force_upstream='auto', nonstream_to_stream=False)
        if 'claude' in result:
            result['claude'].setdefault('stream', {})['nonstream_to_stream'] = False
    from aetherstream.routing.policy_config import global_policy, validate_config
    validate_config(config, global_policy(lambda key: result.get(key)))
    return result, report
