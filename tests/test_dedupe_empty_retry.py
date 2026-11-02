import asyncio
import copy
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from aetherstream.runtime.shared import SharedRuntimeState
from aetherstream.streaming.dedupe import ExactRequestCoalescer


def answer(text='', reason='stop', **message_fields):
    return {'choices': [{'message': {'content': text, **message_fields}, 'finish_reason': reason}],
            'usage': {'completion_tokens': 2725}}


EMPTY = answer(reasoning_content='synthetic test reasoning, not an answer')
GOOD = answer('fresh response')


async def run(coalescer, runner, trace='test'):
    return await coalescer.run(dedupe_key='same', trace_id=trace, upstream_label='fixture', runner=runner)


def empty_child(directory, queue):
    async def work():
        state = SharedRuntimeState(directory)
        coalescer = ExactRequestCoalescer(ttl=180, log=lambda _: None, shared_state=state)
        async def runner():
            with (Path(directory) / 'calls').open('a') as file:
                file.write('call\n')
            for _ in range(500):
                if state.response_waiters('same') == 2:
                    return copy.deepcopy(EMPTY)
                await asyncio.sleep(.01)
            raise RuntimeError('second waiter did not join')
        queue.put(await run(coalescer, runner))
    asyncio.run(work())


class DedupeEmptyRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_result_does_not_block_intentional_retry(self):
        calls = 0
        coalescer = ExactRequestCoalescer(ttl=180, log=lambda _: None)
        async def runner():
            nonlocal calls
            calls += 1
            return copy.deepcopy(EMPTY if calls == 1 else GOOD)
        self.assertEqual(await run(coalescer, runner), (EMPTY, False))
        self.assertEqual(await coalescer.state(), {'inflight': 0, 'recent': 0})
        self.assertEqual(await run(coalescer, runner), (GOOD, False))
        self.assertEqual(calls, 2)

    async def test_concurrent_empty_waiters_share_but_next_request_runs(self):
        gate = asyncio.Event()
        coalescer = ExactRequestCoalescer(ttl=180, log=lambda _: None)
        calls = 0
        async def runner():
            nonlocal calls
            calls += 1
            await gate.wait()
            return copy.deepcopy(EMPTY)
        tasks = [asyncio.create_task(run(coalescer, runner, str(i))) for i in range(3)]
        while not coalescer._inflight or coalescer._inflight['same']['waiters'] < 3:
            await asyncio.sleep(.001)
        gate.set()
        results = await asyncio.gather(*tasks)
        self.assertEqual(calls, 1)
        self.assertEqual([shared for _, shared in results], [False, True, True])
        self.assertTrue(all(result == EMPTY for result, _ in results))
        self.assertEqual(await run(coalescer, runner), (EMPTY, False))
        self.assertEqual(calls, 2)

    async def test_rejected_and_incomplete_results_are_not_recent(self):
        cases = [answer(' ', reasoning_content='reasoning'), answer('partial', 'length'),
                 answer('partial', 'content_filter'), answer('partial', 'refusal'),
                 answer('partial', 'early_stop'), answer(refusal='declined'),
                 {'error': {'message': 'fixture'}}, {'choices': []},
                 {'content': '', 'raw': '{}', 'response': EMPTY},
                 {'collected': ['', 'fixture', {}, 'stop', '{}', []]}]
        for result in cases:
            with self.subTest(result=result):
                calls = 0
                coalescer = ExactRequestCoalescer(ttl=180, log=lambda _: None)
                async def runner():
                    nonlocal calls
                    calls += 1
                    return copy.deepcopy(result)
                await run(coalescer, runner)
                await run(coalescer, runner)
                self.assertEqual(calls, 2)

    async def test_success_and_tool_only_results_still_replay(self):
        tool = {'id': 'call_fixture', 'type': 'function',
                'function': {'name': 'lookup', 'arguments': '{}'}}
        cases = [GOOD, answer(None, 'tool_calls', tool_calls=[tool]),
                 {'content': '', 'raw': '{}', 'response': answer(None, 'tool_calls', tool_calls=[tool])},
                 {'collected': ['', 'fixture', {}, 'tool_use', '{}', [tool]]}]
        for result in cases:
            with self.subTest(result=result):
                calls = 0
                coalescer = ExactRequestCoalescer(ttl=180, log=lambda _: None)
                async def runner():
                    nonlocal calls
                    calls += 1
                    return copy.deepcopy(result)
                self.assertEqual(await run(coalescer, runner), (result, False))
                self.assertEqual(await run(coalescer, runner), (result, True))
                self.assertEqual(calls, 1)

    async def test_old_local_empty_cache_is_ignored(self):
        import time
        coalescer = ExactRequestCoalescer(ttl=180, log=lambda _: None)
        coalescer._recent['same'] = {'payload': EMPTY, 'stored_at': time.monotonic()}
        async def runner():
            return copy.deepcopy(GOOD)
        self.assertEqual(await run(coalescer, runner), (GOOD, False))


class SharedEmptyRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.a = SharedRuntimeState(self.temp.name)
        self.b = SharedRuntimeState(self.temp.name)

    def coalescer(self, state):
        return ExactRequestCoalescer(ttl=180, log=lambda _: None, shared_state=state)

    async def test_empty_result_removed_after_last_waiter_then_fresh_retry(self):
        first, second = self.coalescer(self.a), self.coalescer(self.b)
        async def empty(): return copy.deepcopy(EMPTY)
        async def good(): return copy.deepcopy(GOOD)
        self.assertEqual(await run(first, empty), (EMPTY, False))
        self.assertIsNone(self.a._get('same'))
        self.assertEqual(await run(second, good), (GOOD, False))
        self.assertEqual(self.a._get('same'), GOOD)

    async def test_old_release_empty_cache_does_not_survive_new_cohort(self):
        self.a._set('same', EMPTY, 180)
        async def good(): return copy.deepcopy(GOOD)
        self.assertEqual(await run(self.coalescer(self.b), good), (GOOD, False))

    async def test_crashed_waiter_does_not_pin_empty_cache(self):
        from diskcache import Cache
        self.a._set('same', EMPTY, 180)
        with Cache(self.a.directory) as cache:
            cache.set('response-waiters:same', ['crashed-token'])
        async def good(): return copy.deepcopy(GOOD)
        self.assertEqual(await run(self.coalescer(self.b), good), (GOOD, False))
        self.assertEqual(self.a.response_waiters('same'), 0)

    async def test_live_old_release_waiter_keeps_failed_result_shared(self):
        old_lease = self.a.join_response('same')
        self.a._set('same', EMPTY, 180)
        async def forbidden():
            self.fail('a follower must not automatically send a second request')
        try:
            self.assertEqual(await run(self.coalescer(self.b), forbidden), (EMPTY, True))
            self.assertEqual(self.a._get('same'), EMPTY)
        finally:
            self.a.leave_response('same', old_lease)
        self.assertIsNone(self.a._get('same'))

    def test_concurrent_processes_share_empty_without_automatic_retry(self):
        queue = multiprocessing.Queue()
        processes = [multiprocessing.Process(target=empty_child, args=(self.temp.name, queue)) for _ in range(2)]
        for process in processes: process.start()
        for process in processes:
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(2)
                self.fail('fixture process did not terminate')
            self.assertEqual(process.exitcode, 0)
        results = [queue.get(timeout=2), queue.get(timeout=2)]
        self.assertEqual(sorted(shared for _, shared in results), [False, True])
        self.assertTrue(all(result == EMPTY for result, _ in results))
        self.assertEqual((Path(self.temp.name) / 'calls').read_text(), 'call\n')
        self.assertIsNone(self.a._get('same'))


class ResponseCachePolicyTests(unittest.TestCase):
    def test_native_messages_and_malformed_content(self):
        from aetherstream.runtime.response_cache import response_can_replay
        self.assertFalse(response_can_replay({'type': 'message', 'stop_reason': 'end_turn',
                                             'content': [{'type': 'thinking', 'thinking': 'fixture'}]}))
        self.assertTrue(response_can_replay({'type': 'message', 'stop_reason': 'tool_use',
                                            'content': [{'type': 'tool_use', 'name': 'lookup'}]}))
        self.assertFalse(response_can_replay({'type': 'message', 'stop_reason': 'refusal',
                                             'content': [{'type': 'text', 'text': 'declined'}]}))
        self.assertTrue(response_can_replay(answer([{'type': 'text', 'text': 'visible'}])))
        self.assertFalse(response_can_replay(answer([{'type': 'reasoning', 'text': 'fixture'}])))
        self.assertFalse(response_can_replay(answer(tool_calls=[None, {'function': 'bad'}])))
        self.assertFalse(response_can_replay({'choices': [None]}))
        self.assertTrue(response_can_replay({'oauth_recovery': 'opaque non-inference result'}))


if __name__ == '__main__':
    unittest.main()
