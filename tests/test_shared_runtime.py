import asyncio
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from aetherstream.runtime.shared import SharedRuntimeState
from aetherstream.streaming.dedupe import ExactRequestCoalescer
from aetherstream.streaming.state import ActiveStreamRegistry


def child_response(directory, queue):
    async def work():
        state=SharedRuntimeState(directory)
        async def runner():
            with (Path(directory)/'calls').open('a') as f: f.write('call\n')
            await asyncio.sleep(.3)
            return {'result':'only-once'}
        queue.put(await state.exact_response('key',60,runner))
    asyncio.run(work())


class SharedRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.a=SharedRuntimeState(self.temp.name);self.b=SharedRuntimeState(self.temp.name)

    def test_unknown_state_schema_fails_before_serving(self):
        from diskcache import Cache
        with Cache(self.a.directory) as cache:
            cache.set('__state_schema__',2)
        with self.assertRaisesRegex(RuntimeError,'schema'):
            SharedRuntimeState(self.temp.name)

    def test_session_is_shared_and_preserves_sliding_ttl(self):
        sid,mode=self.a.session('caller',100,now=1000)
        self.assertEqual(mode,'new')
        self.assertEqual(self.b.session('caller',100,now=1050),(sid,'reused'))
        self.assertEqual(self.a.session('caller',100,now=1149),(sid,'reused'))
        self.assertNotEqual(self.b.session('caller',100,now=1250)[0],sid)

    def test_refusal_retires_only_matching_session_across_instances(self):
        old, _ = self.a.session('claude:opus', 100, now=1000)
        other, _ = self.a.session('claude:sonnet', 100, now=1000)
        self.assertTrue(self.b.retire_session('claude:opus', old))
        new, mode = self.a.session('claude:opus', 100, now=1001)
        self.assertEqual(mode, 'new')
        self.assertNotEqual(new, old)
        self.assertFalse(self.b.retire_session('claude:opus', old))
        self.assertEqual(self.b.session('claude:opus', 100, now=1002), (new, 'reused'))
        self.assertEqual(self.b.session('claude:sonnet', 100, now=1002), (other, 'reused'))

    def test_exact_result_is_shared_across_processes(self):
        queue=multiprocessing.Queue()
        ps=[multiprocessing.Process(target=child_response,args=(self.temp.name,queue)) for _ in range(2)]
        for p in ps:p.start()
        for p in ps:p.join(10);self.assertEqual(p.exitcode,0)
        results=[queue.get(timeout=2),queue.get(timeout=2)]
        self.assertEqual(sorted(shared for result,shared in results),[False,True])
        self.assertEqual((Path(self.temp.name)/'calls').read_text(),'call\n')

    async def test_failed_runner_does_not_cache_or_retain_lock(self):
        async def bad():raise ValueError('fixture')
        with self.assertRaises(ValueError):await self.a.exact_response('key',10,bad)
        async def good():return {'good':True}
        self.assertEqual(await self.b.exact_response('key',10,good),({'good':True},False))

    async def test_disconnect_does_not_hide_detached_upstream_task(self):
        coalescer=ExactRequestCoalescer(ttl=60,log=lambda _:None,shared_state=self.a)
        started=asyncio.Event();finish=asyncio.Event()
        async def runner():started.set();await finish.wait();return {'ok':True}
        waiter=asyncio.create_task(coalescer.run(dedupe_key='key',trace_id='x',upstream_label='fixture',runner=runner))
        await started.wait();waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):await waiter
        self.assertEqual((await coalescer.state())['inflight'],1)
        finish.set()
        for _ in range(100):
            if (await coalescer.state())['inflight']==0:break
            await asyncio.sleep(.01)
        self.assertEqual((await coalescer.state())['inflight'],0)
        async def forbidden():self.fail('must use shared result')
        other=ExactRequestCoalescer(ttl=60,log=lambda _:None,shared_state=self.b)
        self.assertEqual(await other.run(dedupe_key='key',trace_id='y',upstream_label='fixture',runner=forbidden),({'ok':True},True))

    async def test_detached_task_barrier_survives_returning_to_caller(self):
        from aetherstream.runtime.tasks import spawn_detached, pending_count
        gate=asyncio.Event()
        task=spawn_detached(gate.wait())
        self.assertEqual(pending_count(),1)
        gate.set();await task
        self.assertEqual(pending_count(),0)

    async def test_superseded_cache_warmup_never_calls_upstream(self):
        from aetherstream.upstreams.anthropic_messages import cache
        state=cache._ClaudeCachePostKeepaliveState(token='old')
        cache._CLAUDE_CACHE_POST_KEEPALIVE_TASKS['fixture']=state
        self.a.replace_token('claude-post:fixture','old')
        self.b.replace_token('claude-post:fixture','new')
        with patch.object(cache,'_run_claude_cache_keepalive_once') as upstream:
            await cache._run_claude_cache_post_keepalive_after_delay(
                post_key='fixture',delay_sec=30,url='http://offline.invalid',request_data={},
                headers={},model='claude-fixture',trace_id='old',
                deps=SimpleNamespace(log=lambda _:None,shared_runtime_state=self.a),
                max_tokens=1,first_data_timeout_sec=1,close_after_data_events=1,max_runs=1,post_state=state)
            upstream.assert_not_called()
        self.assertEqual(self.b.current_token('claude-post:fixture'),'new')
        self.assertNotIn('fixture',cache._CLAUDE_CACHE_POST_KEEPALIVE_TASKS)

    async def test_latest_wins_cross_release_but_other_model_does_not_cancel(self):
        a=ActiveStreamRegistry(log=lambda _:None,shared_state=self.a)
        b=ActiveStreamRegistry(log=lambda _:None,shared_state=self.b)
        try:
            a.register('caller',trace_id='old',model='glm-5.2-local',msg_count=1)
            old=a.cancellation_event('caller','old')
            self.assertEqual(b.get('caller')['trace_id'],'old')
            b.register('caller',trace_id='new',model='glm-5.2-local',msg_count=1,supersede_previous=True)
            await asyncio.wait_for(old.wait(),2)
            new=b.cancellation_event('caller','new')
            a.release_trace('old')
            self.assertEqual(a.get('caller')['trace_id'],'new')
            a.register('caller',trace_id='other',model='claude-fable-5',msg_count=1)
            await asyncio.sleep(.3)
            self.assertFalse(new.is_set())
        finally:
            for obj in (a,b):
                for trace in list(obj._shared_traces):obj.release_trace(trace)
            await asyncio.sleep(.02)
        self.assertEqual(len(a._monitors)+len(b._monitors),0)


if __name__=='__main__':unittest.main()
