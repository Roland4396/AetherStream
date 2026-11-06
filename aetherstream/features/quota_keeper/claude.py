"""Validated model selection, independently switchable from Argon."""
import re
from .client import MODEL


def model_name(value):
    if (not isinstance(value, str) or not re.fullmatch(
            r'claude-(?:opus|sonnet|haiku)-[a-z0-9][a-z0-9_.-]{0,100}', value)):
        raise ValueError('Invalid Claude model')
    return value


def settings(value):
    if value is None:
        return {'enabled': True, 'model': MODEL}
    if not isinstance(value, dict) or set(value) - {'enabled', 'model'}:
        raise ValueError('Invalid Claude settings')
    enabled = value.get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError('Invalid Claude enabled flag')
    return {'enabled': enabled, 'model': model_name(value.get('model', MODEL))}
