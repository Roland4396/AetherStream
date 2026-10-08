"""Side-effect-free readiness and authenticated release diagnostics."""
import asyncio
import json

from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse

from aetherstream.api.quota_keeper_routes import authorized


def register_routes(app, *, runtime, token_file, diagnostics=False):
    @app.get('/ready')
    async def ready():
        state = runtime.ready()
        return JSONResponse(state, status_code=200 if state['ready'] else 503,
                            headers={'X-Stream-Instance': runtime.instance})

    @app.get('/admin/runtime')
    async def status(request: Request):
        if not authorized(request, token_file):
            return JSONResponse({'error': 'Unauthorized'}, status_code=401)
        return JSONResponse(runtime.status())

    if diagnostics:
        @app.get('/admin/runtime/probe/stream')
        async def stream_probe(request: Request, count: int = 20, interval_ms: int = 250):
            if not authorized(request, token_file):
                return JSONResponse({'error': 'Unauthorized'}, status_code=401)
            count, interval_ms = min(120, max(1, count)), min(500, max(10, interval_ms))
            async def chunks():
                for index in range(count):
                    yield 'data: ' + json.dumps({'instance': runtime.instance, 'seq': index}) + '\n\n'
                    await asyncio.sleep(interval_ms / 1000)
                yield 'data: [DONE]\n\n'
            return StreamingResponse(chunks(), media_type='text/event-stream',
                                     headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

        @app.websocket('/admin/runtime/probe/ws')
        async def websocket_probe(socket: WebSocket):
            if not authorized(socket, token_file):
                await socket.close(code=1008)
                return
            await socket.accept()
            try:
                while True:
                    text = await asyncio.wait_for(socket.receive_text(), 60)
                    await socket.send_json({'instance': runtime.instance, 'echo': text})
            except Exception:
                with __import__('contextlib').suppress(Exception):
                    await socket.close()
