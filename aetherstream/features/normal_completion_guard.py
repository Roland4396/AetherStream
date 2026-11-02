"""One explicit retry after a selected provider's consecutive natural finishes.

Stream owns this policy. Pool selection and inference retries remain untouched.
Only protocol adapters can attest an actual upstream terminal; downstream DONE,
proxy cuts, cached replays and cancellation are never evidence of completion.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json

import anyio

from aetherstream.routing.account_channels import ChannelUnavailable
from aetherstream.routing.channel_policy import termination_reason


_OBSERVATION: ContextVar[object | None] = ContextVar('normal_completion_observation', default=None)
NORMAL_FINISH_REASONS = frozenset({'stop', 'end_turn', 'stop_sequence', 'length', 'max_tokens', 'tool_calls', 'tool_use', 'completed'})


class CompletionGuardWarning(Exception):
    def __init__(self, model: str, threshold: int):
        super().__init__(f'HiModels 的模型 {model} 已连续 {threshold} 次正常结束（未触发代理早停）。'
                         '本次请求先暂停警告，尚未发送上游；请再点一次重试以继续。')


class _Observation:
    def __init__(self, account_ids):
        self.account_ids = frozenset(account_ids)
        self.selected = False
        self.reason = None
        self.invalid = False


def observe_selected_account(response) -> None:
    observation = _OBSERVATION.get()
    if observation is not None:
        account = str(getattr(response, 'headers', {}).get('x-account-pool-id') or '')
        observation.selected = account in observation.account_ids


def observe_natural_finish(reason, *, error: bool = False) -> None:
    observation = _OBSERVATION.get()
    if observation is not None:
        observation.reason = reason
        observation.invalid |= error


def invalidate_natural_finish() -> None:
    observation = _OBSERVATION.get()
    if observation is not None:
        observation.invalid = True


class NormalCompletionGuard:
    def __init__(self, shared_state):
        self.shared_state = shared_state

    def _settings(self, plan):
        cfg = plan['policy'].get('runtime', {}).get('normal_completion_guard', {})
        if cfg.get('enabled') is not True:
            return None
        if self.shared_state is None:
            raise ChannelUnavailable('Normal-completion guard requires persistent shared state; no inference was sent')
        # Pool supplies the canonical model; aliases and transport routes share
        # the count, while other channels/providers have independent policy.
        key = json.dumps([cfg['scope'], plan['channel'], plan['upstream_model']], separators=(',', ':'))
        return cfg, key

    def check(self, plan, *, trace_id: str, log) -> None:
        settings = self._settings(plan)
        if settings is None:
            return
        cfg, key = settings
        if self.shared_state.claim_completion_warning(key, cfg['threshold']):
            log(f'[TRACE {trace_id}] normal_completion_guard warning model={plan["model"]} '
                f'threshold={cfg["threshold"]} upstream_sent=false retry_required=true')
            raise CompletionGuardWarning(plan['model'], cfg['threshold'])

    @contextmanager
    def operation(self, plan, *, trace_id: str, log):
        settings = self._settings(plan)
        if settings is None:
            yield
            return
        cfg, key = settings
        observation = _Observation(cfg['account_ids'])
        token = _OBSERVATION.set(observation)
        completed = False
        try:
            yield
            completed = True
        except GeneratorExit:
            # Public Responses/Anthropic converters close their source at a
            # genuine terminal instead of pulling one more item. That closure
            # is still natural only when the adapter attested an upstream end;
            # closing before an end (including client disconnect) resets it.
            completed = observation.reason in NORMAL_FINISH_REASONS
            raise
        finally:
            try:
                if observation.selected:
                    normal = bool(completed and not observation.invalid and not termination_reason()
                                  and observation.reason in NORMAL_FINISH_REASONS)
                    streak = self.shared_state.record_completion(key, normal, cfg['threshold'])
                    log(f'[TRACE {trace_id}] normal_completion_guard outcome model={plan["model"]} '
                        f'normal={str(normal).lower()} streak={streak} threshold={cfg["threshold"]}')
            finally:
                _OBSERVATION.reset(token)

    async def stream(self, body, plan, *, trace_id: str, log):
        with self.operation(plan, trace_id=trace_id, log=log):
            try:
                async for chunk in body:
                    yield chunk
            finally:
                with anyio.CancelScope(shield=True):
                    await body.aclose()
