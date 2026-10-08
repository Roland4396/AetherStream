import asyncio
import importlib.util
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aetherstream.runtime.lifecycle import RuntimeLifecycle, LifecycleMiddleware, FEATURE_CONTRACTS, CONTRACT
from aetherstream.observability.logging import ProxyLogger
from aetherstream.features.claude_replay import ReplayStore


class Feature:
    def __init__(self):
        self.running = False
        self.starts = 0
        self.stops = 0
        self.stop_gate = None
    async def start(self):
        self.running = True
        self.starts += 1
    async def stop(self):
        if self.stop_gate:
            await self.stop_gate.wait()
        self.running = False
        self.stops += 1
    def status(self):
        return {'worker_running': self.running}


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.control=self.root/'active.json'
        self.logs=[];self.instances=[]
    def create(self,slot):
        obj=RuntimeLifecycle(instance=slot,control_file=str(self.control),state_dir=str(self.root),log=self.logs.append,poll_sec=.01)
        obj.register_background('quota_keeper',Feature());self.instances.append(obj);return obj
    async def asyncTearDown(self):
        for obj in self.instances:await obj.stop()
    async def wait_until(self,fn):
        for _ in range(100):
            if fn():return
            await asyncio.sleep(.01)
        self.fail('condition not reached')
    def activate(self,slot):
        self.control.write_text(json.dumps({'active':slot,'state_schema':1}))
    async def test_missing_control_and_standby_never_run_jobs(self):
        a=self.create('blue');await a.start();await asyncio.sleep(.03)
        self.assertFalse(a.owner);self.assertFalse(a.features['quota_keeper'].running)
        self.activate('green');await asyncio.sleep(.03);self.assertFalse(a.owner)
    async def test_exactly_one_owner_handoff_waits_for_old_job(self):
        a=self.create('blue');b=self.create('green');self.activate('blue')
        await a.start();await b.start();await self.wait_until(lambda:a.owner)
        gate=asyncio.Event();a.features['quota_keeper'].stop_gate=gate
        self.activate('green');await asyncio.sleep(.04)
        self.assertTrue(a.owner);self.assertFalse(b.owner)
        gate.set();await self.wait_until(lambda:b.owner)
        self.assertFalse(a.owner);self.assertFalse(a.features['quota_keeper'].running)
        self.assertTrue(b.features['quota_keeper'].running)
    async def test_same_identity_still_cannot_create_two_owners(self):
        a=self.create('blue');b=self.create('blue');self.activate('blue')
        await a.start();await b.start();await asyncio.sleep(.04)
        self.assertEqual(int(a.owner)+int(b.owner),1)
    async def test_malformed_control_relinquishes_jobs(self):
        a=self.create('blue');self.activate('blue');await a.start();await self.wait_until(lambda:a.owner)
        self.control.write_text('{invalid');await self.wait_until(lambda:not a.owner)
    async def test_rollback_reacquires_without_overlapping(self):
        a=self.create('blue');b=self.create('green');self.activate('blue')
        await a.start();await b.start();await self.wait_until(lambda:a.owner)
        self.activate('green');await self.wait_until(lambda:b.owner)
        self.activate('blue');await self.wait_until(lambda:a.owner)
        self.assertFalse(b.owner);self.assertEqual(a.features['quota_keeper'].starts,2)
    async def test_all_http_and_websocket_routes_count_until_cleanup(self):
        obj=self.create('blue');entered=asyncio.Event();release=asyncio.Event()
        async def app(scope,receive,send):
            entered.set();await release.wait()
        middleware=LifecycleMiddleware(app,obj)
        for kind,path in [('http','/future-feature'),('http','/v1/audio/speech'),('websocket','/future-ws')]:
            task=asyncio.create_task(middleware({'type':kind,'path':path,'method':'POST'},None,None))
            await entered.wait();self.assertEqual(len(obj.active),1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
            self.assertEqual(len(obj.active),0);entered.clear()
    async def test_readiness_status_do_not_count_as_work_or_start_jobs(self):
        obj=self.create('blue')
        async def app(scope,receive,send):self.assertEqual(len(obj.active),0)
        middleware=LifecycleMiddleware(app,obj)
        for path in ['/health','/ready','/admin/runtime','/admin/quota-keeper']:
            await middleware({'type':'http','path':path,'method':'GET'},None,None)
        self.assertFalse(obj.owner)
    async def test_exception_cleanup_releases_activity(self):
        obj=self.create('blue')
        async def app(*args):raise ValueError('test')
        with self.assertRaises(ValueError):await LifecycleMiddleware(app,obj)({'type':'http','path':'/new','method':'GET'},None,None)
        self.assertEqual(len(obj.active),0)
    async def test_unknown_background_feature_refused(self):
        with self.assertRaises(ValueError):self.create('blue').register_background('unregistered',Feature())
    async def test_invalid_new_config_fails_readiness(self):
        obj=self.create('blue')
        obj.readiness_check=lambda:(_ for _ in ()).throw(ValueError())
        self.assertFalse(obj.ready()['ready'])
        with self.assertRaises(ValueError):await obj.start()
    async def test_failed_feature_start_releases_lease_and_is_retried(self):
        obj=self.create('blue');feature=obj.features['quota_keeper']
        real_start=feature.start;attempts=[]
        async def start():
            attempts.append(1)
            if len(attempts)==1:raise RuntimeError('fixture')
            await real_start()
        feature.start=start
        self.activate('blue');await obj.start()
        await self.wait_until(lambda:feature.running)
        self.assertGreaterEqual(len(attempts),2)
        self.assertEqual(feature.stops,1)
        self.assertTrue(obj.owner)
        self.assertIsNone(obj.error)
    async def test_pending_tasks_are_visible_even_without_open_connections(self):
        obj=self.create('blue');obj.register_drain_barrier('fixture',lambda:1)
        self.assertEqual(obj.status()['active_http'],0)
        self.assertEqual(obj.status()['pending_tasks'],{'fixture':1})
    def test_every_raw_async_task_has_reviewed_ownership(self):
        import ast
        from collections import Counter
        from aetherstream.runtime.lifecycle import TASK_CONTRACTS
        root=Path(__file__).resolve().parents[1]/'aetherstream'
        found=Counter()
        for file in root.rglob('*.py'):
            class Visitor(ast.NodeVisitor):
                scope=[]
                def visit_FunctionDef(self,node):
                    self.scope.append(node.name);self.generic_visit(node);self.scope.pop()
                visit_AsyncFunctionDef=visit_FunctionDef
                def visit_Call(self,node):
                    name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                    if name in {'create_task','ensure_future'}:
                        found[(str(file.relative_to(root)),'.'.join(self.scope))]+=1
                    self.generic_visit(node)
            Visitor().visit(ast.parse(file.read_text()))
        self.assertEqual(dict(found),{key:value[0] for key,value in TASK_CONTRACTS.items()})
    def test_every_feature_has_declared_execution_scope(self):
        root=Path(__file__).resolve().parents[1]/'aetherstream/features'
        modules={p.stem for p in root.glob('*.py') if p.stem!='__init__'}
        modules|={p.name for p in root.iterdir() if p.is_dir() and (p/'__init__.py').exists()}
        self.assertEqual(modules,set(FEATURE_CONTRACTS))


def log_writer(directory,offset):
    logger=ProxyLogger(debug=False,log_dir=directory)
    for index in range(30):
        trace=str(offset+index)
        logger.save_request_log(model='model-'+trace,messages=[],response=trace,stream=False,trace_id=trace)


def replay_claim(directory,queue):
    store=ReplayStore(log_dir=directory,control_file=str(Path(directory)/'control.json'),log=lambda _:None)
    queue.put(store.consume_if_needed({'mode':'once','raw_sse_path':str(Path(directory)/'01_raw_sse.txt')}))


class SharedStateTests(unittest.TestCase):
    def test_two_processes_keep_log_triples_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            processes=[multiprocessing.Process(target=log_writer,args=(directory,i*100)) for i in range(2)]
            for p in processes:p.start()
            for p in processes:p.join(20);self.assertEqual(p.exitcode,0)
            for p in Path(directory).glob('*_input.json'):
                data=json.loads(p.read_text());out=p.with_name(p.name.replace('_input.json','_output.txt')).read_text()
                self.assertIn('Model: '+data['model']+'\n',out)
                self.assertIn('TraceId: '+data['trace_id']+'\n',out)
    def test_once_replay_claim_is_cross_process_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'01_raw_sse.txt').write_text('fixture')
            (root/'control.json').write_text(json.dumps({'enabled':True,'mode':'once','raw_sse_path':'01_raw_sse.txt'}))
            queue=multiprocessing.Queue()
            ps=[multiprocessing.Process(target=replay_claim,args=(directory,queue)) for _ in range(2)]
            for p in ps:p.start()
            for p in ps:p.join(10);self.assertEqual(p.exitcode,0)
            self.assertEqual(sorted([queue.get(),queue.get()]),[False,True])


class ReleaseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'tools/release.py'
        spec=importlib.util.spec_from_file_location('release_cli',path)
        cls.cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.cli)
    def test_incompatible_state_blocks_rollout(self):
        self.assertTrue(self.cli.compatible(CONTRACT,CONTRACT))
        for field,value in [('state_write',2),('state_read_min',2),('api',2),('lifecycle',2)]:
            self.assertFalse(self.cli.compatible(CONTRACT,{**CONTRACT,field:value}))
    def test_timeout_never_stops_busy_old_container(self):
        with patch.object(self.cli,'command') as command,patch.object(self.cli,'event'):
            self.assertFalse(self.cli.drain({'draining':'blue'},0))
            command.assert_not_called()
    def test_recovery_before_reload_converges_route_without_stopping_active(self):
        journal={'state_schema':1,'phase':'switching','active':'blue','target':'green','old_nginx_workers':['42']}
        with patch.object(self.cli,'read_journal',return_value=journal), \
             patch.object(self.cli,'entry_ready',return_value={'instance':'blue'}), \
             patch.object(self.cli,'switch_route') as route, \
             patch.object(self.cli,'atomic'),patch.object(self.cli,'drain',return_value=True) as drain:
            self.assertTrue(self.cli.recover(1))
            route.assert_called_once_with('blue')
            self.assertEqual(journal['active'],'blue')
            self.assertEqual(journal['draining'],'green')
            drain.assert_called_once_with(journal,1)
    def test_recovery_after_old_stop_is_idempotent(self):
        journal={'active':'green','phase':'draining','draining':'blue'}
        with patch.object(self.cli,'inspect',return_value={'State':{'Running':False}}), \
             patch.object(self.cli,'atomic'),patch.object(self.cli,'command') as command:
            self.assertTrue(self.cli.drain(journal,1))
            command.assert_not_called()
            self.assertEqual(journal['phase'],'stable')
    def test_broad_backend_listener_is_rejected(self):
        from types import SimpleNamespace
        info={'State':{'Pid':123},'NetworkSettings':{'Ports':{}}}
        with patch.object(self.cli,'inspect',return_value=info), \
             patch.object(self.cli,'command',return_value=SimpleNamespace(stdout='LISTEN 0 2048 0.0.0.0:3002 0.0.0.0:*')):
            with self.assertRaisesRegex(RuntimeError,'private'):
                self.cli.validate_candidate_network('blue')
    def test_private_backend_listener_is_accepted(self):
        from types import SimpleNamespace
        info={'State':{'Pid':123},'NetworkSettings':{'Ports':{}}}
        with patch.object(self.cli,'inspect',return_value=info), \
             patch.object(self.cli,'command',return_value=SimpleNamespace(stdout='LISTEN 0 2048 172.28.16.11:3002 0.0.0.0:*')):
            self.cli.validate_candidate_network('blue')
    def test_nginx_never_retries_inference_or_forces_worker_shutdown(self):
        text=(Path(__file__).resolve().parents[1]/'deploy/nginx/nginx.conf').read_text()
        active='\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
        self.assertIn('proxy_next_upstream off;',active)
        self.assertIn('proxy_buffering off;',active)
        self.assertIn('proxy_request_buffering off;',active)
        self.assertNotIn('worker_shutdown_timeout',active)


if __name__=='__main__':unittest.main()
