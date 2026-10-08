"""Offline outbound-payload checks; never sends a prompt to a real provider."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aetherstream.features.opus_notes import (
    PRO_OPUS_LAST_USER_APPEND_TEXT,
    PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER,
)
from aetherstream.features.request_injections import apply_forced_opus_note
from aetherstream.features.stage_warning import STAGE_INTRO, WARNING_OPEN, WARNING_CLOSE
from aetherstream.routing.model_policy import OPENAI_SUBSCRIPTION_MODELS


class GptRequestInjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_subscription_models_both_entrypoints_and_stream_modes(self):
        from starlette.requests import Request
        from aetherstream.api import app
        from aetherstream.api.chat_routes import chat_completions
        from aetherstream.api.protocol_gateway import route_protocol_request

        for model in sorted(OPENAI_SUBSCRIPTION_MODELS):
            for streaming in (False, True):
                for protocol in ('chat', 'responses'):
                    for already_injected in (False, True):
                        with self.subTest(model=model,stream=streaming,protocol=protocol,already_injected=already_injected):
                            deps = copy.copy(app.chat_route_dependencies)
                            # Direct handler tests bypass ASGI cleanup; do not mutate the global registry.
                            deps.active_stream_registry = type(app.active_stream_registry)(log=lambda _: None)
                            deps.release_active_stream_caller = deps.active_stream_registry.release
                            captured = []; logs = []
                            deps.log = logs.append
                            deps.save_request_log = lambda *a, **k: None
                            deps.replay_service = SimpleNamespace(prepare=lambda **k: None)
                            deps.resolve_openai_compatible_route = AsyncMock(return_value=None)
                            deps.RESPONSES_API_KEY = 'offline-test-only'
                            deps.RESPONSES_BASE_URL = 'http://offline.invalid/v1'
                            deps.RESPONSES_REASONING_EFFORT = 'xhigh'
                            deps.GPT_USE_RESPONSES = True
                            deps.GPT_SERVICE_TIER = ''
                            deps.GPT_PROMPT_CACHE_RETENTION = ''
                            # Existing system-policy injection is outside this change.
                            # Use a neutral stub rather than duplicating its production text.
                            deps.inject_gpt_usage_policies_system_message = copy.deepcopy

                            async def fake_stream(**kw):
                                captured.append(copy.deepcopy(kw['request_data']))
                                yield 'data: [DONE]\n\n'

                            async def fake_collect(**kw):
                                captured.append(copy.deepcopy(kw['request_data']))
                                return ('ok', {}, 'stop')

                            deps.forward_responses_as_chat_stream = fake_stream
                            deps.collect_responses_as_chat_completion = fake_collect
                            stage = '<stage><act name="小林">now: 走进车站</act></stage>'
                            text = '<latest_human_message>继续。</latest_human_message>\n' + STAGE_INTRO + '\n' + stage
                            base = {'messages': [
                                {'role':'system','content':'Keep this existing instruction.'},
                                {'role':'user','content':[{'type':'text','text':text}]},
                                {'role':'assistant','content':'Existing prefill, unchanged.'},
                            ]}
                            if already_injected:
                                apply_forced_opus_note(base,selected_model=model,trace_prefix='test',route_label='test',log=lambda _:None)
                            payload = {'model':model,'stream':streaming,'reasoning_effort':'low','messages':base['messages']}
                            if protocol == 'responses':
                                payload['input'] = payload.pop('messages')
                                for item in payload['input']:
                                    if isinstance(item['content'],list):
                                        for block in item['content']: block['type'] = 'input_text'
                            original = copy.deepcopy(payload)
                            path = '/v1/responses' if protocol == 'responses' else '/v1/chat/completions'
                            request = Request({'type':'http','method':'POST','path':path,'headers':[],
                                               'client':('127.0.0.1',1234)})
                            request._body = json.dumps(payload).encode()
                            with patch('httpx.AsyncClient.send', side_effect=AssertionError('External network forbidden')):
                                response = (await route_protocol_request(request,deps,protocol='responses')
                                            if protocol == 'responses' else await chat_completions(request,deps))
                                if hasattr(response,'body_iterator'):
                                    async for _ in response.body_iterator: pass
                            self.assertEqual(response.status_code,200,logs)
                            self.assertEqual(len(captured),1,logs)
                            outgoing = captured[0]
                            def content_text(content):
                                if isinstance(content,str): return content
                                return '\n'.join(block.get('text','') for block in content if isinstance(block,dict))
                            body = '\n'.join(content_text(item.get('content',[])) for item in outgoing['input']
                                             if item.get('role')=='user')
                            self.assertEqual(body.count(PRO_OPUS_LAST_USER_APPEND_TEXT),1)
                            self.assertEqual(body.count(PRO_OPUS_LAST_USER_ILLUSTRATION_MARKER),3)
                            self.assertIn(WARNING_OPEN+stage+WARNING_CLOSE,body)
                            self.assertNotIn('<disclaimer>',body)
                            self.assertIn('Keep this existing instruction.',outgoing['instructions'])
                            self.assertEqual(outgoing['input'][-1]['role'],'assistant')
                            self.assertEqual(outgoing['model'],model)
                            self.assertEqual(outgoing['reasoning']['effort'],'xhigh')
                            self.assertEqual(payload,original)
                            self.assertTrue(any('gpt_last_user_note' in line for line in logs))


if __name__ == '__main__':
    unittest.main()
