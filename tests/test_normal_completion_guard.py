"""No paid inference: provider terminals, warning-once and cross-process state."""
import asyncio
import copy
import json
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import httpx

from aetherstream.features.normal_completion_guard import (
    NormalCompletionGuard, CompletionGuardWarning, observe_selected_account,
    observe_natural_finish, invalidate_natural_finish,
)
from aetherstream.routing.channel_policy import policy_scope, mark_termination
from aetherstream.routing.policy_config import validate_policy, RegistryError
from aetherstream.runtime.shared import SharedRuntimeState
import test_channel_gateway as gateway_tests
from test_channel_gateway import plan, events, ControlledStream, chat_json


def guarded_plan(model='alias-a', upstream='canonical-a', transport='chat'):
    value = plan(model=model, transport=transport)
    value['upstream_model'] = upstream
    value['policy']['runtime']['normal_completion_guard'] = {'enabled': True, 'scope': 'himodels',
                                                  'threshold': 2, 'account_ids': ['himodels_1']}
    return value


def claim_warning(directory, key, queue):
    queue.put(SharedRuntimeState(directory).claim_completion_warning(key, 2))


class CompletionGuardStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.state = SharedRuntimeState(self.temp.name)
        self.guard = NormalCompletionGuard(self.state)
        self.logs = []
        self.target = guarded_plan()

    def record(self, reason='stop', account='himodels_1', terminate=False, invalid=False, failed=False):
        with policy_scope(self.target['policy']):
            try:
                with self.guard.operation(self.target, trace_id='offline', log=self.logs.append):
                    observe_selected_account(SimpleNamespace(headers={'x-account-pool-id': account}))
                    observe_natural_finish(reason)
                    if terminate: mark_termination('early_stop')
                    if invalid: invalidate_natural_finish()
                    if failed: raise RuntimeError('offline')
            except RuntimeError:
                pass

    def check(self, target=None):
        self.guard.check(target or self.target, trace_id='offline', log=self.logs.append)

    def test_two_natural_finishes_warn_exactly_once_then_repeat_cycle(self):
        for _ in range(2): self.check(); self.record()
        with self.assertRaises(CompletionGuardWarning): self.check()
        self.check()
        self.record(); self.check(); self.record()
        with self.assertRaises(CompletionGuardWarning): self.check()
        self.assertIn('upstream_sent=false', self.logs[-1])

    def test_all_non_normal_outcomes_break_the_streak(self):
        for change in ({'terminate': True}, {'reason': None}, {'invalid': True}, {'failed': True},
                       {'reason': 'refusal'}, {'reason': 'content_filter'}):
            with self.subTest(change=change):
                self.record(); self.record(**change); self.record(); self.check()
                self.record(terminate=True)

    def test_other_providers_do_not_count_or_reset_this_provider(self):
        self.record(account='other'); self.record(account='other'); self.check()
        self.record(); self.record(account='other'); self.record()
        with self.assertRaises(CompletionGuardWarning): self.check()

    def test_aliases_share_canonical_model_other_models_channels_do_not(self):
        self.record(); self.record()
        self.check(guarded_plan(upstream='canonical-b'))
        other = guarded_plan(); other['channel'] = 'other-channel'; self.check(other)
        with self.assertRaises(CompletionGuardWarning): self.check(guarded_plan(model='alias-b'))
        self.check()

    def test_state_survives_new_instance_and_uses_additive_schema_one_keys(self):
        session, _ = self.state.session('existing', 600)
        self.record(); self.record()
        restarted = NormalCompletionGuard(SharedRuntimeState(self.temp.name))
        with self.assertRaises(CompletionGuardWarning): restarted.check(self.target, trace_id='new', log=self.logs.append)
        self.assertEqual(SharedRuntimeState(self.temp.name).session('existing', 600)[0], session)

    def test_concurrent_cross_process_claims_produce_one_warning_only(self):
        self.state.record_completion('offline-key', True, 2)
        self.state.record_completion('offline-key', True, 2)
        ctx = multiprocessing.get_context('fork'); queue = ctx.Queue()
        workers = [ctx.Process(target=claim_warning, args=(self.temp.name, 'offline-key', queue)) for _ in range(6)]
        for worker in workers: worker.start()
        for worker in workers: worker.join(10); self.assertEqual(worker.exitcode, 0)
        self.assertEqual(sum(queue.get(timeout=2) for _ in workers), 1)

    def test_no_shared_store_fails_closed_only_when_enabled(self):
        guard = NormalCompletionGuard(None)
        disabled = guarded_plan(); disabled['policy']['runtime'].pop('normal_completion_guard')
        guard.check(disabled, trace_id='offline', log=self.logs.append)
        from aetherstream.routing.account_channels import ChannelUnavailable
        with self.assertRaises(ChannelUnavailable): guard.check(self.target, trace_id='offline', log=self.logs.append)

    def test_policy_validation_rejects_missing_wrong_or_unknown_fields(self):
        validate_policy(self.target['policy'])
        for change in ({'threshold': True}, {'threshold': 0}, {'enabled': 'yes'}, {'account_ids': []},
                       {'scope': ''}, {'automatic_retry': True}):
            value = copy.deepcopy(self.target['policy']); value['runtime']['normal_completion_guard'].update(change)
            with self.assertRaises(RegistryError): validate_policy(value)
        value = copy.deepcopy(self.target['policy']); del value['runtime']['normal_completion_guard']['account_ids']
        with self.assertRaises(RegistryError): validate_policy(value)


class CompletionGuardGatewayTests(unittest.IsolatedAsyncioTestCase):
    invoke = gateway_tests.ChannelGatewayTests.invoke

    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.guard = NormalCompletionGuard(SharedRuntimeState(self.temp.name))

    def deps(self, resolved):
        deps = gateway_tests.ChannelGatewayTests.deps(self, resolved)
        deps.normal_completion_guard = self.guard
        return deps

    async def complete(self, target=None, *, account='himodels_1', downstream=None):
        resolved = target or guarded_plan()
        result = chat_json('wire-model'); result['choices'][0]['finish_reason'] = 'stop'
        return await self.invoke(resolved, {'model': resolved['model'], 'messages': [], 'stream': False},
            lambda _: httpx.Response(200, headers={'x-account-pool-id': account}, json=result), downstream=downstream)

    async def test_warning_is_http_409_and_sends_no_inference_retry_is_explicit(self):
        for _ in range(2):
            response, _, sent, _ = await self.complete()
            self.assertEqual(response.status_code, 200); self.assertEqual(len(sent), 1)
        response, body, sent, _ = await self.complete(guarded_plan(model='alias-b'))
        self.assertEqual(response.status_code, 409)
        error = json.loads(body)['error']
        self.assertTrue(error['retry_required']); self.assertFalse(error['upstream_sent'])
        self.assertEqual(sent, [])
        response, _, sent, _ = await self.complete()
        self.assertEqual(response.status_code, 200); self.assertEqual(len(sent), 1)

    async def test_native_chat_and_messages_terminals_count_all_public_protocols(self):
        for transport in ('chat', 'messages'):
            for downstream in (None, 'responses', 'anthropic'):
                with self.subTest(transport=transport, downstream=downstream):
                    resolved = guarded_plan(upstream=f'{transport}-{downstream}', transport=transport)
                    for _ in range(2):
                        upstream = ControlledStream(events(transport))
                        response, body, sent, _ = await self.invoke(resolved,
                            {'model': resolved['model'], 'messages': [], 'input': 'Synthetic', 'stream': True},
                            lambda _: httpx.Response(200, headers={'content-type': 'text/event-stream', 'x-account-pool-id': 'himodels_1'}, stream=upstream),
                            downstream=downstream)
                        self.assertEqual(response.status_code, 200, body); self.assertEqual(len(sent), 1)
                        self.assertTrue(upstream.closed)
                    response, body, sent, _ = await self.invoke(resolved,
                        {'model': resolved['model'], 'messages': [], 'input': 'Synthetic', 'stream': True},
                        lambda _: self.fail('warning must not open an upstream'), downstream=downstream)
                    self.assertEqual(response.status_code, 409, body); self.assertEqual(sent, [])

    async def test_proxy_cut_resets_streak_in_sse_and_json_collection(self):
        for transport in ('chat', 'messages'):
            for stream in (True, False):
                resolved = guarded_plan(upstream=f'cut-{transport}-{stream}', transport=transport)
                resolved['policy']['streaming']['force_upstream'] = 'stream'
                for stop in (False, True, False, False):
                    upstream = ControlledStream(events(transport, stop=stop), wait_after=stop)
                    response, body, sent, _ = await asyncio.wait_for(self.invoke(resolved,
                        {'model': resolved['model'], 'messages': [], 'stream': stream},
                        lambda _: httpx.Response(200, headers={'content-type': 'text/event-stream', 'x-account-pool-id': 'himodels_1'}, stream=upstream)), 3)
                    self.assertEqual(response.status_code, 200, body); self.assertEqual(len(sent), 1)
                    self.assertTrue(upstream.closed)
                with self.assertRaises(CompletionGuardWarning): self.guard.check(resolved, trace_id='offline', log=lambda _: None)

    async def test_http_error_and_incomplete_stream_are_not_normal_completion(self):
        resolved = guarded_plan()
        await self.complete(resolved)
        for status in (503, 200):
            chunks = [] if status != 200 else [b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n']
            upstream = ControlledStream(chunks)
            _, _, sent, _ = await self.invoke(resolved,
                {'model': resolved['model'], 'messages': [], 'stream': True},
                lambda _: httpx.Response(status, headers={'content-type': 'text/event-stream', 'x-account-pool-id': 'himodels_1'}, stream=upstream))
            self.assertEqual(len(sent), 1)
        await self.complete(resolved)  # Only one natural completion after the failure.
        self.guard.check(resolved, trace_id='offline', log=lambda _: None)

    async def test_downstream_close_before_terminal_breaks_streak_and_closes_socket(self):
        from aetherstream.api.channel_gateway import chat_completions
        from test_channel_gateway import request, REAL_CLIENT
        from unittest.mock import patch
        resolved = guarded_plan()
        await self.complete(resolved)
        upstream = ControlledStream(events('chat', stop=False)[:2], wait_after=True)
        deps = self.deps(resolved)
        def factory(*_, **__):
            return REAL_CLIENT(transport=httpx.MockTransport(lambda _: httpx.Response(200,
                headers={'content-type': 'text/event-stream', 'x-account-pool-id': 'himodels_1'}, stream=upstream)))
        # Keep the fake transport alive through lazy iteration and cleanup.
        with patch.object(httpx, 'AsyncClient', side_effect=factory):
            response = await chat_completions(request({'model': resolved['model'], 'messages': [], 'stream': True}), deps)
            # ASGI iterates and closes in the same request context. wait_for
            # creates separate Tasks and would invalidate ContextVar tokens.
            async with asyncio.timeout(2):
                while b'OK' not in await anext(response.body_iterator):
                    pass
                await response.body_iterator.aclose()
        self.assertTrue(upstream.closed); self.assertEqual(len(deps.active_stream_registry), 0)
        await self.complete(resolved)
        self.guard.check(resolved, trace_id='offline', log=lambda _: None)

    async def test_true_messages_json_counts_but_proxy_truncation_does_not(self):
        resolved = guarded_plan(transport='messages')
        for text in ('OK', 'visible<CUT>ignored', 'OK', 'OK'):
            result = {'type': 'message', 'id': 'offline', 'model': 'wire-model',
                      'content': [{'type': 'text', 'text': text}], 'stop_reason': 'end_turn',
                      'usage': {'input_tokens': 4, 'output_tokens': 2}}
            response, body, sent, _ = await self.invoke(resolved,
                {'model': resolved['model'], 'messages': [], 'stream': False},
                lambda _: httpx.Response(200, headers={'x-account-pool-id': 'himodels_1'}, json=result))
            self.assertEqual(response.status_code, 200, body); self.assertEqual(len(sent), 1)
        with self.assertRaises(CompletionGuardWarning): self.guard.check(resolved, trace_id='offline', log=lambda _: None)

    async def test_cached_exact_response_counts_the_actual_runner_only(self):
        from aetherstream.api.channel_gateway import chat_completions
        from test_channel_gateway import request, REAL_CLIENT
        from unittest.mock import patch
        resolved = guarded_plan(); resolved['policy']['streaming']['exact_dedupe'] = True
        deps = self.deps(resolved)
        from unittest.mock import AsyncMock
        deps.build_account_dedupe = AsyncMock(return_value={'key': 'offline-key', 'shape': 'chat'})
        cached = None
        async def once(**kwargs):
            nonlocal cached
            if cached is None:
                cached = await kwargs['runner'](); return cached, False
            return cached, True
        deps.run_exact_nonstream_once = once
        calls = []
        def upstream(req):
            calls.append(req); result = chat_json('wire'); result['choices'][0]['finish_reason'] = 'stop'
            return httpx.Response(200, headers={'x-account-pool-id': 'himodels_1'}, json=result)
        def factory(*_, **__): return REAL_CLIENT(transport=httpx.MockTransport(upstream))
        with patch.object(httpx, 'AsyncClient', side_effect=factory):
            for _ in range(3):
                response = await chat_completions(request({'model': resolved['model'], 'messages': [], 'stream': False}), deps)
                self.assertEqual(response.status_code, 200)
        self.assertEqual(len(calls), 1)
        self.guard.check(resolved, trace_id='offline', log=lambda _: None)


if __name__ == '__main__': unittest.main()
