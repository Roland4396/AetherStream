"""Last live waiter owns cancellation, across instances and OS processes."""
import asyncio,multiprocessing,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from aetherstream.runtime.shared import SharedRuntimeState
from aetherstream.streaming.dedupe import ExactRequestCoalescer


def orphan(directory):
    state=SharedRuntimeState(directory)
    lease=state.join_response('key')
    (Path(directory)/'joined').write_text('yes')
    os._exit(0)  # Kernel unlock, deliberately no Python finally.


def shared_process(directory, role):
    root=Path(directory)
    async def run():
        state=SharedRuntimeState(directory+'/state')
        c=ExactRequestCoalescer(ttl=60,log=lambda _:None,shared_state=state)
        async def runner():
            with (root/'calls').open('a') as f:f.write(role+'\n')
            (root/'started').touch()
            try:
                while not (root/'release').exists():await asyncio.sleep(.005)
                return {'ok':True}
            finally:(root/'closed').touch()
        task=asyncio.create_task(c.run(dedupe_key='process-key',trace_id=role,upstream_label='fake',runner=runner))
        while not task.done() and not (root/('cancel-'+role)).exists():await asyncio.sleep(.005)
        if not task.done():task.cancel()
        try:
            result=await task
            assert result[0]=={'ok':True}
        except asyncio.CancelledError:pass
        (root/('client-left-'+role)).touch()
        while c._inflight:await asyncio.sleep(.005)
    asyncio.run(run())


def until(predicate, timeout=5):
    limit=time.monotonic()+timeout
    while not predicate():
        if time.monotonic()>limit:raise AssertionError('Timed out waiting for isolated process fixture')
        time.sleep(.01)


class DedupeCancellationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.shared=SharedRuntimeState(self.temp.name)

    def coalescer(self, shared=True):
        return ExactRequestCoalescer(ttl=60,log=lambda _:None,
             shared_state=SharedRuntimeState(self.temp.name) if shared else None)

    async def pair(self, cross, cancel_first, finish):
        a=self.coalescer(shared=cross);b=self.coalescer() if cross else a
        started,closed,release=asyncio.Event(),asyncio.Event(),asyncio.Event();calls=0
        async def runner():
            nonlocal calls
            calls+=1;started.set()
            try:await release.wait();return {'ok':True}
            finally:closed.set()
        one=asyncio.create_task(a.run(dedupe_key='key',trace_id='a',upstream_label='fake',runner=runner))
        await started.wait()
        two=asyncio.create_task(b.run(dedupe_key='key',trace_id='b',upstream_label='fake',runner=runner))
        for _ in range(100):
            if (self.shared.response_waiters('key')==2 if cross else a._inflight['key']['waiters']==2):break
            await asyncio.sleep(.01)
        stopped,live=(one,two) if cancel_first else (two,one)
        stopped.cancel()
        with self.assertRaises(asyncio.CancelledError):await stopped
        await asyncio.sleep(.06)
        self.assertFalse(closed.is_set(),'Another client still owns the request')
        self.assertEqual(calls,1)
        if finish:
            release.set();result=await asyncio.wait_for(live,2)
            self.assertEqual(result[0],{'ok':True})
        else:
            live.cancel()
            with self.assertRaises(asyncio.CancelledError):await live
            await asyncio.wait_for(closed.wait(),1)
        for _ in range(100):
            if not a._inflight and not b._inflight:break
            await asyncio.sleep(.01)
        self.assertFalse(a._inflight);self.assertFalse(b._inflight)
        self.assertEqual(calls,1)
        self.assertEqual(self.shared.response_waiters('key'),0)

    async def test_local_leader_leaves_follower_finishes(self):await self.pair(False,True,True)
    async def test_local_follower_leaves_leader_finishes(self):await self.pair(False,False,True)
    async def test_local_last_follower_cancels(self):await self.pair(False,True,False)
    async def test_remote_leader_leaves_follower_finishes(self):await self.pair(True,True,True)
    async def test_remote_follower_leaves_leader_finishes(self):await self.pair(True,False,True)
    async def test_remote_last_follower_cancels_original_owner(self):await self.pair(True,True,False)

    def test_waiter_locks_do_not_expire_and_crashed_process_is_pruned(self):
        lease=self.shared.join_response('key')
        self.assertEqual(SharedRuntimeState(self.temp.name).response_waiters('key'),1)
        child=multiprocessing.Process(target=orphan,args=(self.temp.name,));child.start();child.join(5)
        self.assertEqual(child.exitcode,0)
        self.assertEqual(self.shared.response_waiters('key'),1)
        self.shared.leave_response('key',lease)
        self.assertEqual(self.shared.response_waiters('key'),0)
        self.assertEqual(list(self.shared.waiter_locks.iterdir()),[])

    def process_pair(self, complete):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);state=SharedRuntimeState(directory+'/state')
            ctx=multiprocessing.get_context('spawn')
            a=ctx.Process(target=shared_process,args=(directory,'owner'))
            b=ctx.Process(target=shared_process,args=(directory,'follower'))
            children=[]
            try:
                a.start();children.append(a);until(lambda:(root/'started').exists())
                b.start();children.append(b);until(lambda:state.response_waiters('process-key')==2)
                (root/'cancel-owner').touch();until(lambda:(root/'client-left-owner').exists())
                time.sleep(.07)
                self.assertFalse((root/'closed').exists(),'Remote follower still needs the actual upstream')
                self.assertEqual((root/'calls').read_text().splitlines(),['owner'])
                then=time.monotonic()
                (root/('release' if complete else 'cancel-follower')).touch()
                until(lambda:(root/'closed').exists())
                self.assertLess(time.monotonic()-then,.5)
                for process in children:process.join(5);self.assertEqual(process.exitcode,0)
                self.assertEqual(state.response_waiters('process-key'),0)
                self.assertEqual((root/'calls').read_text().splitlines(),['owner'])
                if not complete:self.assertIsNone(state._get('process-key'))
            finally:
                for process in children:
                    if process.is_alive():process.terminate();process.join(2)

    def test_real_process_owner_cancel_follower_finishes(self):self.process_pair(True)
    def test_real_process_last_follower_cancel_closes_owner(self):self.process_pair(False)

    async def test_cancel_before_worker_first_step_keeps_no_barrier_or_waiter(self):
        coalescer=self.coalescer()
        original=asyncio.create_task
        def cancel_before_start(coro):
            task=original(coro)
            task.cancel()
            return task
        async def forbidden():self.fail('upstream cannot begin')
        with patch('aetherstream.streaming.dedupe.asyncio.create_task',cancel_before_start):
            with self.assertRaises(asyncio.CancelledError):
                await coalescer.run(dedupe_key='key',trace_id='pre-start',upstream_label='fake',runner=forbidden)
        self.assertEqual(await coalescer.state(),{'inflight':0,'recent':0})
        self.assertEqual(self.shared.response_waiters('key'),0)

    async def test_new_client_waits_for_previous_socket_cleanup(self):
        coalescer=self.coalescer();started,closing,release=asyncio.Event(),asyncio.Event(),asyncio.Event()
        async def old():
            started.set()
            try:await asyncio.Event().wait()
            finally:closing.set();await release.wait()
        one=asyncio.create_task(coalescer.run(dedupe_key='key',trace_id='old',upstream_label='fake',runner=old))
        await started.wait();one.cancel();await closing.wait()
        self.assertEqual(len(coalescer._inflight),1)
        seen=[]
        async def new():seen.append(True);return {'new':True}
        two=asyncio.create_task(coalescer.run(dedupe_key='key',trace_id='new',upstream_label='fake',runner=new))
        await asyncio.sleep(.05);self.assertEqual(seen,[])
        release.set()
        with self.assertRaises(asyncio.CancelledError):await one
        self.assertEqual((await asyncio.wait_for(two,2))[0],{'new':True})
        self.assertEqual(seen,[True]);self.assertFalse(coalescer._inflight)

if __name__=='__main__':unittest.main()
