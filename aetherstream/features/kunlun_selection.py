"""Additive, configuration-owned Kunlun model-family selection.

The existing exact-name preset configuration stays readable by older releases.
Optional rules live at the runtime root, which older readers safely ignore.
This module selects an existing envelope; it never edits the preset or routes.
"""
from __future__ import annotations

import copy
import re

from aetherstream.features.kunlun_linux import selected_settings as exact_settings
from aetherstream.routing.policy_config import matches, validate_selector


def validate_model_rules(rules):
    if rules is None:
        return
    if not isinstance(rules, list):
        raise ValueError('Kunlun model rules must be a list')
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) - {'match', 'channels'}:
            raise ValueError('Invalid Kunlun model rule')
        if not isinstance(rule.get('match'), dict) or not rule['match']:
            raise ValueError('Kunlun model rules require a nonempty selector')
        try:
            validate_selector(rule['match'])
        except re.error as exc:
            raise ValueError('Invalid Kunlun model regular expression') from exc
        channels = rule.get('channels')
        if channels is not None and (
                not isinstance(channels, list) or not channels
                or not all(isinstance(value, str) and value for value in channels)):
            raise ValueError('Kunlun rule channels must be nonempty channel names')


def selected_settings(config, selection, *, model_rules=None):
    validate_model_rules(model_rules)
    selected = exact_settings(config, selection)
    if selected is not None:
        return selected
    if (config is None or not config.get('enabled')
            or selection.get('transport') not in config.get('transports', ['chat'])):
        return None
    names = [selection.get(key) for key in ('model', 'upstream_model')]
    for rule in model_rules or []:
        if 'channels' in rule and selection.get('channel') not in rule['channels']:
            continue
        if any(isinstance(name, str) and matches(name, rule['match']) for name in names):
            return copy.deepcopy(config)
    return None
