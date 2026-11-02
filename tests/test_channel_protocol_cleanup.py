"""Native completion and pre-body cancellation must release channel ownership."""
import asyncio
import unittest

from aetherstream.routing.channel_policy import policy_stream, current_policy
from aetherstream.transforms.downstream_protocols import chat_stream_to_anthropic, chat_stream_to_responses
from test_channel_gateway import events, plan, policy
import test_channel_gateway as gateway


class ProtocolCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_protocol_terminal_closes_source_in_the_request_context(self):
        for convert in (chat_stream_to_anthropic, chat_stream_to_responses):
            closed = []
            settings = policy()

            async def source():
                try:
                    for part in events('chat'):
                        yield part
                    await asyncio.Event().wait()
                finally:
                    closed.append(current_policy())

            result = [part async for part in convert(policy_stream(source(), settings), model='test')]
            self.assertTrue(result)
            self.assertEqual(closed, [settings])
            self.assertIsNone(current_policy())

    async def test_cancel_on_protocol_start_releases_not_yet_iterated_channel(self):
        for protocol in ('anthropic', 'responses'):
            response, _, sent, deps = await gateway.ChannelGatewayTests().invoke(
                plan(), {'model': 'free/test', 'stream': True, 'messages': []},
                lambda _: self.fail('provider must not start'), consume=False, downstream=protocol)
            self.assertEqual(len(deps.active_stream_registry), 1)

            async def send(message):
                if message['type'] == 'http.response.body':
                    raise asyncio.CancelledError()

            with self.assertRaises(asyncio.CancelledError):
                await response.stream_response(send)
            self.assertEqual(sent, [])
            self.assertEqual(len(deps.active_stream_registry), 0)


if __name__ == '__main__':
    unittest.main()
