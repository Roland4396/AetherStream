"""Release lifecycle contract shared by all request and background features.

One host / local shared filesystem: kernel-backed FileLock is the ownership
boundary. A multi-host release must replace this with fenced distributed leases.
"""
import asyncio
from contextlib import suppress
import json
from pathlib import Path
import time
import uuid

from filelock import FileLock, Timeout

# Adding a feature requires declaring its execution/state scope and tests.
FEATURE_CONTRACTS = {
    'claude_replay': 'shared-file', 'drawing_filter': 'request',
    'early_stop': 'request', 'glm_thinking': 'request', 'gproxy_oauth': 'request',
    'gpt_policy': 'request', 'kimi_sampling': 'request',
    'opus_notes': 'request', 'pro_compat': 'request', 'quota_keeper': 'background',
    'replay': 'request', 'request_injections': 'request', 'scene_filter': 'request',
    'stage_warning': 'request', 'terminal_tool': 'request',
}
CONTRACT = {'lifecycle': 1, 'api': 1, 'state_read_min': 1, 'state_read_max': 1, 'state_write': 1}


class RuntimeLifecycle:
    def __init__(self, *, instance: str, control_file: str, state_dir: str, log,
                 readiness_check=lambda: None, poll_sec=1):
        self.instance = instance
        self.control_file = Path(control_file) if control_file else None
        self.state_dir = Path(state_dir)
        self.readiness_check = readiness_check
        self.log = log
        self.poll_sec = poll_sec
        self.features = {}
        self.barriers = {}
        self.active = {}
        self.started = False
        self.owner = False
        self.error = None
        self._features_started = []
        self._lease = None
        self._task = None
        self._stop = asyncio.Event()

    def register_background(self, name, feature):
        if self.started or name in self.features or FEATURE_CONTRACTS.get(name) != 'background':
            raise ValueError('Background feature is unregistered or registration is late/duplicate')
        if not all(callable(getattr(feature, method, None)) for method in ('start', 'stop', 'status')):
            raise TypeError('Background features must implement start, stop and status')
        self.features[name] = feature

    def register_drain_barrier(self, name, pending_count):
        if self.started or name in self.barriers:
            raise ValueError('Duplicate or late drain barrier')
        self.barriers[name] = pending_count

    def desired_owner(self):
        if self.control_file is None:
            return True  # standalone dev/test mode still takes the shared lease
        try:
            value = json.loads(self.control_file.read_text())
            return value['active'] == self.instance and value['state_schema'] == CONTRACT['state_write']
        except (OSError, ValueError, KeyError, TypeError):
            return False  # missing/malformed control never authorizes jobs

    async def start(self):
        if self.started:
            return
        self.readiness_check()
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lease = FileLock(self.state_dir / 'background.lock', thread_local=False, mode=0o600)
        self.started = True
        self._stop.clear()
        self._task = asyncio.create_task(self._supervise(), name='stream-feature-supervisor')

    async def _release(self):
        # Never relinquish ownership before every feature has stopped. Failure
        # keeps the lease held and blocks promotion, instead of creating 2 owners.
        for name in reversed(self._features_started.copy()):
            await self.features[name].stop()
            self._features_started.remove(name)
        if self.owner:
            self._lease.release()
            self.owner = False
            self.log(f'runtime_lifecycle ownership_released instance={self.instance}')

    async def _supervise(self):
        try:
            while not self._stop.is_set():
                try:
                    desired = self.desired_owner()
                    if self.owner and not desired:
                        await self._release()
                    if desired and not self.owner:
                        try:
                            self._lease.acquire(timeout=0)
                        except Timeout:
                            pass
                        else:
                            self.owner = True
                            try:
                                for name, feature in self.features.items():
                                    self._features_started.append(name)
                                    await feature.start()
                            except BaseException:
                                await self._release()
                                raise
                            self.log(f'runtime_lifecycle ownership_acquired instance={self.instance}')
                    self.error = None
                except Exception as exc:
                    self.error = type(exc).__name__
                    self.log(f'runtime_lifecycle error instance={self.instance} type={self.error}')
                try:
                    await asyncio.wait_for(self._stop.wait(), self.poll_sec)
                except asyncio.TimeoutError:
                    pass
        finally:
            await self._release()

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task
            self._task = None
        self.started = False

    def enter(self, kind, path):
        key = uuid.uuid4().hex
        self.active[key] = {'kind': kind, 'path': path, 'started_at': time.time()}
        return key

    def leave(self, key):
        self.active.pop(key, None)

    def ready(self):
        try:
            self.readiness_check()
            valid = self.started
            error = None
        except Exception as exc:
            valid, error = False, type(exc).__name__
        return {'ready': valid, 'instance': self.instance, 'contract': CONTRACT, 'error': error}

    def status(self):
        desired = self.desired_owner()
        return {**self.ready(), 'phase': 'serving' if desired else 'draining' if self.active else 'standby',
                'background_owner': self.owner, 'background_error': self.error,
                'background_features': {name: feature.status() for name, feature in self.features.items()},
                'active_http': sum(r['kind'] == 'http' for r in self.active.values()),
                'active_websockets': sum(r['kind'] == 'websocket' for r in self.active.values()),
                'pending_tasks': {name: count() for name, count in self.barriers.items()},
                'active_requests': list(self.active.values())}


class LifecycleMiddleware:
    """All ASGI protocols/routes are tracked until final cleanup, not headers."""
    def __init__(self, app, runtime):
        self.app, self.runtime = app, runtime

    async def __call__(self, scope, receive, send):
        kind, path = scope.get('type'), scope.get('path', '')
        probe = kind == 'http' and scope.get('method') in ('GET', 'HEAD') and path in (
            '/health', '/ready', '/admin/runtime', '/admin/quota-keeper')
        if kind not in ('http', 'websocket') or probe:
            return await self.app(scope, receive, send)
        key = self.runtime.enter(kind, path)
        try:
            await self.app(scope, receive, send)
        finally:
            self.runtime.leave(key)

# Raw task creation is reviewed by a regression gate. Request-scoped tasks must
# be awaited/cancelled in finally; detached tasks use runtime.tasks; singleton
# jobs live under this supervisor. Counts make extra tasks in old functions
# visible too, instead of silently allowing them through the same filename.
TASK_CONTRACTS = {
    ('features/quota_keeper/service.py', 'start'): (1, 'singleton-lifecycle'),
    ('runtime/lifecycle.py', 'start'): (1, 'supervisor'),
    ('runtime/tasks.py', 'spawn_detached'): (1, 'detached-drain-barrier'),
    ('streaming/state.py', 'register'): (1, 'stream-cancel-drain-barrier'),
    ('streaming/dedupe.py', 'run'): (1, 'coalescer-drain-barrier'),
    ('streaming/json_keepalive.py', '__call__'): (2, 'request-finally'),
    ('api/audio_routes.py', '_send_until_disconnect'): (2, 'request-finally'),
    ('upstreams/openai_responses.py', 'forward_responses_as_chat_stream'): (1, 'request-finally'),
    ('upstreams/openai_responses.py', 'replay_responses_as_chat_stream'): (1, 'request-finally'),
    ('upstreams/openai_chat_completions.py', 'forward_chat_completions_stream'): (2, 'request-finally'),
    ('upstreams/openai_chat_completions.py', 'replay_chat_completions_nonstream_as_stream'): (1, 'request-finally'),
    ('upstreams/gemini_generate_content.py', 'forward_gemini_generate_content_stream'): (1, 'request-finally'),
    ('upstreams/anthropic_messages/chat_stream.py', 'forward_anthropic_messages_as_chat_stream'): (2, 'request-finally'),
}
