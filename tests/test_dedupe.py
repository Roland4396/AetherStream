import asyncio
import unittest

from aetherstream.streaming.dedupe import ExactRequestCoalescer


class ExactRequestCoalescerTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_and_recent_requests_share_one_result(self):
        calls = 0
        release = asyncio.Event()
        coalescer = ExactRequestCoalescer(ttl=60, log=lambda _message: None)

        async def runner():
            nonlocal calls
            calls += 1
            await release.wait()
            return {"value": calls}

        first = asyncio.create_task(coalescer.run(
            dedupe_key="same",
            trace_id="first",
            upstream_label="test",
            runner=runner,
        ))
        await asyncio.sleep(0)
        second = asyncio.create_task(coalescer.run(
            dedupe_key="same",
            trace_id="second",
            upstream_label="test",
            runner=runner,
        ))
        await asyncio.sleep(0)
        release.set()
        first_result, second_result = await asyncio.gather(first, second)

        recent_result = await coalescer.run(
            dedupe_key="same",
            trace_id="third",
            upstream_label="test",
            runner=runner,
        )
        self.assertEqual(calls, 1)
        self.assertFalse(first_result[1])
        self.assertTrue(second_result[1])
        self.assertTrue(recent_result[1])
        self.assertEqual(recent_result[0], {"value": 1})

    async def test_last_requester_cancellation_closes_upstream_without_cache(self):
        calls = 0
        started, closed = asyncio.Event(), asyncio.Event()
        coalescer = ExactRequestCoalescer(ttl=60, log=lambda _: None)
        async def runner():
            nonlocal calls
            calls += 1
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        requester = asyncio.create_task(coalescer.run(dedupe_key='cancelled', trace_id='one',
                                                      upstream_label='test', runner=runner))
        await started.wait()
        requester.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await requester
        self.assertTrue(closed.is_set())
        self.assertEqual(await coalescer.state(), {'inflight': 0, 'recent': 0})
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
