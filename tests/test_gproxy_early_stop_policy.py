import asyncio
import inspect
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from aetherstream.upstreams import openai_responses as responses
from aetherstream.upstreams import openai_chat_completions as chat
from aetherstream.upstreams.anthropic_messages import chat_stream, collectors, messages
from aetherstream.upstreams.anthropic_messages.types import AnthropicMessagesDeps
from aetherstream.upstreams.route_logging import scope_stop_detection


TAG = '<disclaimer>'
PARTS = ['HEAD', '<dis', 'claimer>', 'TAIL']
TEXT = ''.join(PARTS)


def deps(kind, logs):
    cls = {'responses':responses.ResponsesUpstreamDeps,
           'chat':chat.ChatCompletionsUpstreamDeps,
           'anthropic':AnthropicMessagesDeps}[kind]
    return cls(log=logs.append,save_request_log=lambda *a,**kw:None,
               build_openai_sse_error=lambda *a,**kw:b'UNEXPECTED_ERROR',
               has_stop_tag=lambda s:TAG in s,find_stop_tag=lambda s:s.find(TAG),
               fmt_ms=lambda *_:'0ms',release_caller=lambda *_:None)


class SelectionTests(unittest.TestCase):
    def test_selected_account_not_model_or_requested_pool_controls_detection(self):
        original=deps('responses',[])
        for account,url,disabled in [
            ('gproxy_1','http://account-pool-proxy:3200/v1/responses',True),
            ('gproxy_codex_subscription_1','http://account-pool-proxy:3200/v1/responses',True),
            ('paid_fallback','http://account-pool-proxy:3200/v1/responses',False),
            ('not_gproxy_1','https://other.invalid/v1',False),
            ('','http://gproxy:8787/codex/v1/responses',True),
            ('','http://127.0.0.1:8787/an/v1/messages',True),
            ('','https://not-gproxy.example/v1',False),
        ]:
            with self.subTest(account=account,url=url):
                selected=scope_stop_detection(original,SimpleNamespace(headers={'x-account-pool-id':account}),url,'test ')
                self.assertEqual(selected.has_stop_tag(TEXT),not disabled)
                self.assertEqual(selected.find_stop_tag(TEXT),-1 if disabled else 4)
                self.assertTrue(original.has_stop_tag(TEXT),'Must not mutate shared deps')


def events(kind):
    if kind=='responses':
        data=[{'type':'response.output_text.delta','delta':part} for part in PARTS]
        data.append({'type':'response.completed','response':json_response()})
    elif kind=='chat':
        data=[{'id':'test','model':'test','choices':[{'index':0,'delta':{'content':part},'finish_reason':None}]} for part in PARTS]
        data.append({'choices':[{'index':0,'delta':{},'finish_reason':'stop'}],
                     'usage':{'prompt_tokens':100,'completion_tokens':23,'total_tokens':123}})
    else:
        data=[{'type':'message_start','message':{'id':'test','role':'assistant','model':'test',
                'content':[],'usage':{'input_tokens':100,'output_tokens':0}}},
              {'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}}]
        data += [{'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':part}} for part in PARTS]
        data += [{'type':'content_block_stop','index':0},
                 {'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':{'output_tokens':23}},
                 {'type':'message_stop'}]
    lines=['data: '+json.dumps(x)+'\n\n' for x in data]
    if kind!='anthropic': lines.append('data: [DONE]\n\n')
    return lines


def json_response():
    return {'id':'resp_test','status':'completed','model':'test',
            'output':[{'type':'message','role':'assistant','content':[{'type':'output_text','text':TEXT}]}],
            'usage':{'input_tokens':100,'output_tokens':23,'total_tokens':123}}


def stream_text(chunks):
    content=[]
    for line in b''.join(chunks).decode().splitlines():
        if not line.startswith('data: '):continue
        try:event=json.loads(line[6:])
        except ValueError:continue
        if event.get('type')=='content_block_delta':
            content.append(event.get('delta',{}).get('text',''))
        for choice in event.get('choices',[]):
            content.append(choice.get('delta',{}).get('content',''))
    return ''.join(content)


class GproxyStreamPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_live_transports_read_past_marker_only_for_gproxy(self):
        cases=[
            ('responses',responses.forward_responses_as_chat_stream,False),
            ('responses',responses.collect_responses_as_chat_completion,False),
            ('responses',responses.collect_responses_nonstream,True),
            ('responses',responses.replay_responses_as_chat_stream,True),
            ('chat',chat.forward_chat_completions_stream,False),
            ('chat',chat.collect_chat_completions_stream,False),
            ('anthropic',chat_stream.forward_anthropic_messages_as_chat_stream,False),
            ('anthropic',messages.forward_anthropic_messages_stream,False),
            ('anthropic',collectors.collect_anthropic_messages_as_chat_completion,False),
            ('anthropic',collectors.collect_anthropic_messages_response,False),
        ]
        for kind,fn,is_json in cases:
            for account in ('gproxy_1','gproxy_codex_subscription_1','other_provider'):
                with self.subTest(function=fn.__name__,account=account):
                    logs=[];exhausted=asyncio.Event();d=deps(kind,logs)
                    class Body(httpx.AsyncByteStream):
                        async def __aiter__(self):
                            for event in events(kind):
                                yield event.encode()
                                await asyncio.sleep(0)
                            exhausted.set()
                    def handle(request):
                        headers={'x-account-pool-id':account}
                        return (httpx.Response(200,headers=headers,json=json_response()) if is_json
                                else httpx.Response(200,headers=headers,stream=Body()))
                    client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
                    all_args=dict(url='http://account-pool-proxy:3200/v1/test',api_key='test-only',
                        request_data={'model':'test','stream':True},headers={},model='test',messages=[],
                        trace_id='offline',caller_key='',caller_desc='',timeout=httpx.Timeout(2),
                        max_raw_sse_bytes=8192,deps=d,enable_early_stop=True)
                    args={k:v for k,v in all_args.items() if k in inspect.signature(fn).parameters}
                    with patch('httpx.AsyncClient',return_value=client):
                        async with asyncio.timeout(4):
                            result=fn(**args)
                            if inspect.isasyncgen(result):
                                chunks=[item async for item in result]
                                self.assertNotIn(b'UNEXPECTED_ERROR',b''.join(chunks),logs)
                                text=stream_text(chunks)
                            else:
                                result=await result
                                text=result[0]
                                if isinstance(text,dict):text=''.join(part['text'] for part in text['content'])
                    disabled=account.startswith('gproxy_')
                    if disabled:
                        self.assertEqual(text,TEXT,logs)
                    else:
                        # Existing stream behavior can already have sent a split
                        # marker prefix; this change must not alter that behavior.
                        self.assertTrue(text.startswith('HEAD'),logs)
                        self.assertNotIn('TAIL',text,logs)
                    self.assertEqual(any('early_stop_policy enabled=false' in x for x in logs),disabled,logs)
                    self.assertTrue(d.has_stop_tag(TEXT),'Shared matcher was changed')
                    self.assertTrue(client.is_closed)

    async def test_gproxy_manual_cancellation_still_closes_connection(self):
        waiting=asyncio.Event();closed=asyncio.Event();logs=[]
        class WaitingBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield ('data: '+json.dumps({'type':'response.output_text.delta','delta':TEXT})+'\n\n').encode()
                waiting.set()
                await asyncio.Event().wait()
            async def aclose(self):closed.set()
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,
            headers={'x-account-pool-id':'gproxy_codex_subscription_1'},stream=WaitingBody())))
        async def consume():
            async for _ in responses.forward_responses_as_chat_stream(url='http://account-pool-proxy:3200/v1/responses',
                api_key='test',request_data={'model':'test'},model='test',messages=[],trace_id='cancel',
                caller_key='',caller_desc='',timeout=httpx.Timeout(2),max_raw_sse_bytes=8192,deps=deps('responses',logs)):
                pass
        with patch('httpx.AsyncClient',return_value=client):
            task=asyncio.create_task(consume())
            await asyncio.wait_for(waiting.wait(),2)
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(closed.is_set())
        self.assertTrue(client.is_closed)


if __name__=='__main__':unittest.main()
