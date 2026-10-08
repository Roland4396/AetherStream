import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from aetherstream.features.early_stop import EarlyStopMatcher
from pathlib import Path
from aetherstream.upstreams.openai_chat_completions import (
    ChatCompletionsUpstreamDeps, forward_chat_completions_stream,
)


class KimiEarlyStopTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_tag_closes_unfinished_upstream_and_ignores_reasoning(self):
        tags = json.loads((Path(__file__).resolve().parents[1] / 'runtime-flags.json').read_text())['early_stop']['tags']
        cases = [(tag, False, split) for tag in tags for split in (False, True)]
        cases += [(tag, True, True) for tag in tags]
        for tag, himodels, split in cases:
            parts = ['answer', tag + 'tail'] if not split else ['answer', tag[:len(tag)//2], tag[len(tag)//2:] + 'tail']
            with self.subTest(tag=tag, himodels=himodels, split=split):
                closed = asyncio.Event()
                logs = []
                saved = []

                class Upstream(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        deltas = [{'reasoning_content': '<disclaimer>reasoning is not an answer'}]
                        deltas += [{'content': part} for part in parts]
                        for delta in deltas:
                            event = {'id': 'test', 'model': 'transsion/kimi-k3', 'choices': [
                                {'index': 0, 'delta': delta, 'finish_reason': None},
                            ]}
                            yield ('data: ' + json.dumps(event) + '\n\n').encode()
                            await asyncio.sleep(0)
                        # No finish_reason or DONE: only proxy early stop can finish.
                        await asyncio.Event().wait()

                    async def aclose(self):
                        closed.set()

                matcher = EarlyStopMatcher(lookup=lambda *keys: [tag] if keys == ('early_stop', 'tags') else None, env_enabled=True,
                                           env_tags=None, env_case_sensitive=True)
                deps = ChatCompletionsUpstreamDeps(
                    log=logs.append, save_request_log=lambda *a, **k: saved.append((a, k)),
                    build_openai_sse_error=lambda *a, **k: b'ERROR',
                    has_stop_tag=matcher.has, find_stop_tag=matcher.find,
                    fmt_ms=lambda *_: '0ms', release_caller=lambda *_: None,
                )
                client = httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda req: httpx.Response(200, stream=Upstream(),
                        headers={'x-account-pool-id': 'himodels_1'} if himodels else {})))
                with patch('aetherstream.upstreams.openai_chat_completions.httpx.AsyncClient', return_value=client):
                    async with asyncio.timeout(3):
                        chunks = [part async for part in forward_chat_completions_stream(
                            url='https://test.invalid/v1/chat/completions', headers={},
                            request_data={'model': 'transsion/kimi-k3', 'stream': True},
                            timeout=httpx.Timeout(2), max_raw_sse_bytes=4096, deps=deps,
                            enable_early_stop=not himodels, model='free/kimi-k3' if himodels else 'transsion/kimi-k3', messages=[],
                        )]
                output = b''.join(chunks).decode()
                self.assertTrue(closed.is_set())
                self.assertTrue(client.is_closed)
                self.assertEqual(output.count('data: [DONE]'), 1)
                self.assertIn('answer', output)
                self.assertNotIn('tail', output)
                self.assertTrue(any('reason=early_stop_tag' in log for log in logs), logs)
                self.assertEqual(saved[0][0][2], 'answer')
