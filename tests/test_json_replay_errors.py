"""Synthetic errors only; never contact a provider or load production secrets."""
import asyncio
import json
import unittest
from unittest.mock import Mock

import httpx

from aetherstream.streaming.modes import replay_json_completion
from aetherstream.upstreams.errors import UpstreamHTTPError, public_stream_error


def parse_events(body):
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith(b'data: {')]


class JsonReplayErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_delayed_429_keeps_heartbeats_then_sends_one_error_without_retry(self):
        attempts = 0
        callback = Mock()
        async def operation():
            nonlocal attempts
            attempts += 1
            await asyncio.sleep(.025)
            raise UpstreamHTTPError(429, json.dumps({'error': {'message': 'Quota exhausted',
                'type': 'rate_limit_error', 'code': 'insufficient_quota', 'param': None}}))
        body = b''.join([x async for x in replay_json_completion(
            operation, model='synthetic', heartbeat_seconds=.005, on_error=callback)])
        records = parse_events(body)
        self.assertGreaterEqual(len(records), 3)
        errors = [r['error'] for r in records if 'error' in r]
        self.assertEqual(errors, [{'type': 'rate_limit_error', 'status': 429,
            'code': 'insufficient_quota', 'param': None, 'message': 'Quota exhausted'}])
        self.assertEqual(attempts, 1)
        callback.assert_called_once_with(errors[0], 'UpstreamHTTPError')
        self.assertNotIn(b'[DONE]', body)
        self.assertTrue(all(r['choices'][0]['finish_reason'] is None for r in records if 'choices' in r))

    async def test_timeout_and_transport_errors_are_bounded_and_never_leak_exception_text(self):
        for error, status in ((httpx.ReadTimeout('SECRET URL'), 504),
                              (TimeoutError('SECRET builtin timeout'), 504),
                              (httpx.ConnectError('SECRET credential'), 502),
                              (ValueError('SECRET internal details'), 502)):
            with self.subTest(error=type(error).__name__):
                async def operation():
                    raise error
                async def consume():
                    return b''.join([x async for x in replay_json_completion(operation, model='synthetic', heartbeat_seconds=.005)])
                body = await asyncio.wait_for(consume(), .5)
                self.assertEqual(parse_events(body)[-1]['error']['status'], status)
                self.assertNotIn(b'SECRET', body)
                self.assertNotIn(b'[DONE]', body)

    async def test_error_json_is_preserved_and_observed_once(self):
        value = {'message': 'Synthetic error', 'type': 'invalid_request_error', 'code': 'bad_input'}
        callback = Mock()
        async def operation():
            return {'error': value}
        body = b''.join([x async for x in replay_json_completion(operation, model='synthetic', on_error=callback)])
        self.assertEqual(parse_events(body)[-1], {'error': value})
        callback.assert_called_once_with(value, None)
        self.assertNotIn(b'[DONE]', body)

    async def test_broken_error_logger_does_not_destroy_the_wire_error(self):
        async def operation():
            raise UpstreamHTTPError(503, 'Temporarily unavailable')
        body = b''.join([x async for x in replay_json_completion(
            operation, model='synthetic', on_error=Mock(side_effect=OSError('disk full')))])
        self.assertEqual(parse_events(body)[-1]['error']['status'], 503)

    async def test_close_before_operation_starts_sends_no_upstream_request(self):
        operation = Mock()
        iterator = replay_json_completion(operation, model='synthetic')
        await anext(iterator)
        await iterator.aclose()
        operation.assert_not_called()

    async def test_cancellation_joins_operation_and_is_not_converted_to_an_error(self):
        started, stopped = asyncio.Event(), asyncio.Event()
        callback = Mock()
        async def operation():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                stopped.set()
        iterator = replay_json_completion(operation, model='synthetic', on_error=callback)
        await anext(iterator)
        pending = asyncio.create_task(anext(iterator))
        await started.wait()
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertTrue(stopped.is_set())
        callback.assert_not_called()

    async def test_generator_close_joins_pending_operation(self):
        stopped = asyncio.Event()
        async def operation():
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        iterator = replay_json_completion(operation, model='synthetic', heartbeat_seconds=.005)
        await anext(iterator)
        await anext(iterator)
        await iterator.aclose()
        self.assertTrue(stopped.is_set())

    async def test_close_after_error_yield_does_not_raise_task_error_again(self):
        async def operation():
            raise UpstreamHTTPError(429, 'Quota exhausted')
        iterator = replay_json_completion(operation, model='synthetic')
        await anext(iterator)
        self.assertIn(b'"error"', await anext(iterator))
        await iterator.aclose()

    async def test_success_and_early_stop_contracts_remain_unchanged(self):
        for reason in ('stop', 'early_stop'):
            async def operation():
                return {'choices': [{'message': {'content': 'Synthetic result'}, 'finish_reason': reason}]}
            callback = Mock()
            body = b''.join([x async for x in replay_json_completion(operation, model='synthetic', on_error=callback)])
            self.assertIn(b'Synthetic result', body)
            self.assertEqual(b'[DONE]' in body, reason == 'stop')
            callback.assert_not_called()


class PublicErrorTests(unittest.TestCase):
    def test_non_json_error_and_large_error_are_bounded(self):
        for body in ('Upstream unavailable', 'x' * 100000):
            error = public_stream_error(UpstreamHTTPError(503, body))
            self.assertEqual(error['status'], 503)
            self.assertLessEqual(len(error['message']), 4000)
        self.assertEqual(public_stream_error(UpstreamHTTPError(429, '[]'))['code'], 429)

    def test_httpx_status_error_preserves_status_without_request_url(self):
        request = httpx.Request('POST', 'http://synthetic.invalid/?key=SECRET')
        response = httpx.Response(429, request=request, json={'error': {'message': 'Quota exhausted'}})
        error = public_stream_error(httpx.HTTPStatusError('SECRET URL', request=request, response=response))
        self.assertEqual(error['status'], 429)
        self.assertNotIn('SECRET', json.dumps(error))
