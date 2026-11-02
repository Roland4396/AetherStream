"""Stream consumes Pool selections; it never selects by channel prefix."""
import copy
import unittest
from aetherstream.routing.policy_config import enrich_selection, validate_config, RegistryError, dedupe_identity
from aetherstream.routing.channel_policy import policy_scope
from aetherstream.upstreams.route_logging import _selected_stop_settings
from types import SimpleNamespace


def fixture():
    return {'early_stop':{'enabled':True,'action':'disconnect','tags':['<CUT>']},
            'streaming':{'force_upstream':'auto','nonstream_to_stream':False},
            'claude':{'effort':'medium'},
            'channel_policies':{'schema':1,'defaults':{'injections':{'opus_note':True}},
                'profiles':{'profile':{'injections':{'opus_note':False}}},
                'model_rules':[{'match':{'exact':['vendor/model']},'policy':{'filters':{'scene':True}}}],
                'channels':{'opaque-channel':{'profiles':['profile'],'routes':{'opaque-route':{}}}},
                'account_overrides':{'real-account':{'early_stop':{'enabled':False}}}}}


def selection():
    return {'model':'a-prefix-the-pool-chose/model','upstream_model':'vendor/model',
            'channel':'opaque-channel','route':'opaque-route','transport':'chat','revision':'pool-only'}


class PolicyOwnershipTests(unittest.TestCase):
    def test_channel_math_switch_is_scoped_and_survives_opus_model_rule(self):
        flags = fixture()
        flags['channel_policies']['channels']['anti'] = {'policy': {'injections': {'math_reminder': False}}}
        flags['channel_policies']['model_rules'].append({
            'match': {'regex': 'opus'}, 'policy': {'injections': {'opus_note': True, 'illustration': True}}})
        for channel, expected in [('anti', False), ('another-channel', True)]:
            selected = {**selection(), 'channel': channel, 'upstream_model': 'claude-opus-4-6'}
            plan = enrich_selection(lambda key: flags.get(key), selected)
            self.assertEqual(plan['policy']['injections'].get('math_reminder', True), expected)
            self.assertTrue(plan['policy']['injections']['opus_note'])
            self.assertTrue(plan['policy']['injections']['illustration'])

    def test_pool_selection_is_consumed_without_reselecting_or_rewriting(self):
        flags=fixture();selected=selection()
        plan=enrich_selection(lambda key:flags.get(key),selected)
        for key,value in selected.items():self.assertEqual(plan[key],value)
        self.assertNotIn('policy',selected)
        self.assertFalse(plan['policy']['injections']['opus_note'])
        self.assertTrue(plan['policy']['filters']['scene'])
        self.assertEqual(plan['policy']['runtime']['claude']['effort'],'medium')

    def test_false_and_empty_local_settings_are_not_lost(self):
        flags=fixture()
        flags['channel_policies']['channels']['opaque-channel']['routes']['opaque-route']['policy']={'early_stop':{'enabled':False,'tags':[]}}
        plan=enrich_selection(lambda key:flags.get(key),selection())
        self.assertEqual(plan['policy']['early_stop']['tags'],[])
        self.assertFalse(plan['policy']['early_stop']['enabled'])

    def test_stream_rejects_selection_fields_in_feature_config(self):
        for key in ['match','models','strip_prefix','accounts','transport']:
            flags=fixture();flags['channel_policies']['channels']['opaque-channel']['routes']['opaque-route'][key]='wrong owner'
            with self.assertRaises(RegistryError):enrich_selection(lambda key:flags.get(key),selection())

    def test_new_pool_route_uses_existing_channel_defaults_without_second_registration(self):
        flags=fixture();selected=selection();selected['route']='not-configured'
        plan=enrich_selection(lambda key:flags.get(key),selected)
        self.assertEqual(plan['route'],'not-configured')
        self.assertEqual(plan['transport'],'chat')
        self.assertFalse(plan['policy']['injections']['opus_note'])
        self.assertTrue(plan['policy']['early_stop']['enabled'])
        self.assertTrue(plan['policy']['filters']['scene'])

    def test_new_pool_channel_uses_global_defaults_and_model_rules(self):
        flags=fixture();selected=selection()
        selected.update(channel='new-supplier',route='account-models:new-supplier:chat',
                        policy={'early_stop':{'enabled':False,'tags':['REMOTE']}})
        plan=enrich_selection(lambda key:flags.get(key),selected)
        self.assertEqual(plan['channel'],'new-supplier')
        self.assertEqual(plan['model'],selected['model'])
        self.assertTrue(plan['policy']['injections']['opus_note'])
        self.assertTrue(plan['policy']['early_stop']['enabled'])
        self.assertEqual(plan['policy']['early_stop']['tags'],['<CUT>'])
        self.assertTrue(plan['policy']['filters']['scene'])
        self.assertIn('real-account',plan['policy']['account_overrides'])

    def test_pool_sent_policy_cannot_override_local_feature_settings(self):
        flags=fixture();selected=selection();selected['policy']={'early_stop':{'enabled':False,'tags':['REMOTE']}}
        plan=enrich_selection(lambda key:flags.get(key),selected)
        self.assertEqual(plan['policy']['early_stop']['tags'],['<CUT>'])
        with policy_scope(plan['policy']):
            remote=SimpleNamespace(headers={'x-acc-early-stop':'eyJlbmFibGVkIjpmYWxzZX0=', 'x-account-pool-id':'unconfigured-account'})
            self.assertTrue(_selected_stop_settings(remote)['enabled'])
            remote.headers['x-account-pool-id']='real-account'
            self.assertFalse(_selected_stop_settings(remote)['enabled'])

    def test_trigger_word_hot_change_is_seen_by_next_plan_only(self):
        flags=fixture();first=enrich_selection(lambda key:flags.get(key),selection())
        flags['early_stop']['tags'].append('USER-ADDED')
        second=enrich_selection(lambda key:flags.get(key),selection())
        self.assertEqual(first['policy']['early_stop']['tags'],['<CUT>'])
        self.assertEqual(second['policy']['early_stop']['tags'],['<CUT>','USER-ADDED'])
        self.assertNotEqual(first['policy_revision'],second['policy_revision'])

    def test_invalid_stop_policy_fails_readiness_validation(self):
        for change in [{'action':'hide-output'},{'enabled':'false'},{'tags':'wrong'}]:
            flags=fixture();flags['early_stop'].update(change)
            with self.assertRaises(RegistryError):enrich_selection(lambda key:flags.get(key),selection())
