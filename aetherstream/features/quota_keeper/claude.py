"""Validated model selection, independently switchable from Argon."""
import re
from .client import MODEL, native_id


def model_name(value):
    if (not isinstance(value, str) or not re.fullmatch(
            r'claude-(?:opus|sonnet|haiku)-[a-z0-9][a-z0-9_.-]{0,100}', value)):
        raise ValueError('Invalid Claude model')
    return value


def settings(value):
    if value is None:
        return {'enabled': True, 'model': MODEL, 'account_models': {}}
    if not isinstance(value, dict) or set(value) - {'enabled', 'model', 'account_models'}:
        raise ValueError('Invalid Claude settings')
    enabled = value.get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError('Invalid Claude enabled flag')
    overrides = value.get('account_models', {})
    if not isinstance(overrides, dict):
        raise ValueError('Invalid account model map')
    normalized = {}
    for credential_id, model in overrides.items():
        if not isinstance(credential_id, str):
            raise ValueError('Invalid account model key')
        key = native_id('credentials', credential_id)
        selected = model_name(model)
        if key in normalized and normalized[key] != selected:
            raise ValueError('Conflicting account model overrides')
        normalized[key] = selected
    return {'enabled': enabled, 'model': model_name(value.get('model', MODEL)),
            'account_models': normalized}


def select_model(config, members):
    """Explicit overrides only; duplicate imports of one account must agree."""
    choices = {config.get('account_models', {}).get(native_id('credentials', c['id']))
               for c in members}
    choices.discard(None)
    if len(choices) > 1:
        raise ValueError('Conflicting model selection for duplicate account imports')
    return next(iter(choices)) if choices else config['model']
