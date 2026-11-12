import copy
import unittest

from aetherstream.features.kunlun_linux import SOURCE_SHA256, selected_settings as legacy_select
from aetherstream.features.kunlun_selection import selected_settings, validate_model_rules
from aetherstream.routing.channel_policy import current_policy, policy_scope
from aetherstream.routing.policy_config import RegistryError, dedupe_identity, enrich_selection


CONFIG = {
    'enabled': True,
    'match': {'channels': ['free'], 'public_models': ['free/gemini-3.8-flash-high']},
    'transports': ['chat'], 'source_sha256': SOURCE_SHA256, 'default_insert_count': 3,
}
RULES = [{'match': {'regex': r'gemini-4-(?:argon|carbon)(?:$|[-_/])'}}]


def selection(model='[P]gemini-4-argon', channel='anti', transport='chat', upstream=None):
    return {'model': model, 'upstream_model': upstream or model, 'channel': channel,
            'route': 'unchanged-pool-route', 'transport': transport, 'revision': 'pool-revision'}


def flags(rules=None):
    result = {'channel_policies': {'schema': 1, 'defaults': {}, 'profiles': {},
                                  'model_rules': [], 'channels': {}},
              'kunlun_linux': copy.deepcopy(CONFIG)}
    if rules is not None:
        result['kunlun_linux_model_rules'] = copy.deepcopy(rules)
    return result


def resolve(value, selected):
    return enrich_selection(lambda key: value.get(key), selected)


class KunlunSelectionTests(unittest.TestCase):
    def test_missing_and_empty_rules_preserve_legacy_selection(self):
        for selected in (selection(), selection('free/gemini-3.8-flash-high', 'free'),
                         selection('free/gemini-3.8-flash-high', 'anti'),
                         selection(transport='messages')):
            for rules in (None, []):
                self.assertEqual(selected_settings(CONFIG, selected, model_rules=rules),
                                 legacy_select(CONFIG, selected))

    def test_families_prefixes_and_tiers_are_selected_without_channel_hardcoding(self):
        for channel in ('free', 'anti', 'future-channel'):
            for prefix in ('', 'free/', '[P]', '[官]', '[按量]'):
                for family in ('argon', 'carbon'):
                    for suffix in ('', '-low', '-medium', '-high', '-future-tier'):
                        name = prefix + 'gemini-4-' + family + suffix
                        with self.subTest(channel=channel, model=name):
                            self.assertEqual(selected_settings(CONFIG, selection(name, channel),
                                                               model_rules=RULES), CONFIG)

    def test_canonical_upstream_and_public_names_are_both_checked(self):
        for public, upstream in (('opaque-alias', 'gemini-4-carbon-high'),
                                 ('[P]gemini-4-argon', 'opaque-upstream')):
            self.assertEqual(selected_settings(CONFIG, selection(public, upstream=upstream),
                                               model_rules=RULES), CONFIG)

    def test_case_insensitive_selector_and_future_suffix(self):
        self.assertEqual(selected_settings(CONFIG, selection('[X]GEMINI-4-CARBON-future'),
                                           model_rules=RULES), CONFIG)

    def test_near_names_and_other_models_do_not_match(self):
        for name in ('gemini-4-carbonated', 'gemini-4-argonium', 'gemini-3.8-flash-high',
                     'claude-opus-5-5-low', 'gemini-3.1-pro', 'unrelated-model'):
            with self.subTest(model=name):
                self.assertIsNone(selected_settings(CONFIG, selection(name), model_rules=RULES))

    def test_explicit_rule_channel_scope_is_honored(self):
        rules = [{'match': RULES[0]['match'], 'channels': ['anti']}]
        self.assertEqual(selected_settings(CONFIG, selection(), model_rules=rules), CONFIG)
        self.assertIsNone(selected_settings(CONFIG, selection(channel='other'), model_rules=rules))

    def test_disabled_absent_and_unsupported_transport_never_select(self):
        for config in (None, {**CONFIG, 'enabled': False}):
            self.assertIsNone(selected_settings(config, selection(), model_rules=RULES))
        for transport in ('messages', 'responses', 'gemini', 'local'):
            self.assertIsNone(selected_settings(CONFIG, selection(transport=transport),
                                                model_rules=RULES))

    def test_selectors_use_existing_exact_prefix_any_semantics(self):
        rules = [{'match': {'any': [{'exact': ['custom-exact']}, {'prefix': 'custom/'}]}}]
        for name in ('custom-exact', 'custom/future'):
            self.assertEqual(selected_settings(CONFIG, selection(name), model_rules=rules), CONFIG)
        self.assertIsNone(selected_settings(CONFIG, selection('other'), model_rules=rules))

    def test_invalid_rules_are_rejected_before_requests(self):
        invalid = ({}, '', False, [None], [{}], [{'match': {}}],
                   [{'match': {'regex': '['}}], [{'match': {'unknown': 'x'}}],
                   [{'match': {'exact': 'not-a-list'}}],
                   [{'match': {'prefix': 1}}], [{'match': {'any': []}}],
                   [{'match': RULES[0]['match'], 'channels': []}],
                   [{'match': RULES[0]['match'], 'channels': 'anti'}],
                   [{'match': RULES[0]['match'], 'channels': [None]}],
                   [{'match': RULES[0]['match'], 'accounts': ['not-owned-here']}])
        for value in invalid:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                validate_model_rules(value)
            with self.subTest(policy_value=value), self.assertRaises(RegistryError):
                resolve(flags(value), selection())

    def test_selection_is_a_copy_and_inputs_are_immutable(self):
        config = copy.deepcopy(CONFIG); rules = copy.deepcopy(RULES); selected = selection()
        before = copy.deepcopy((config, rules, selected))
        value = selected_settings(config, selected, model_rules=rules)
        value['match']['public_models'].append('changed-copy')
        self.assertEqual((config, rules, selected), before)

    def test_enrichment_preserves_pool_routes_modes_and_other_feature_fields(self):
        old = flags(); new = flags(RULES); selected = selection()
        a, b = resolve(old, selected), resolve(new, selected)
        for key in selected:
            self.assertEqual(a[key], b[key])
        self.assertEqual(a['request_mode'], b['request_mode'])
        before = copy.deepcopy(b['policy'])
        self.assertEqual(before['runtime'].pop('kunlun_linux'), CONFIG)
        self.assertEqual(before, a['policy'])

    def test_other_models_and_existing_exact_targets_keep_dedupe_identity(self):
        payload = {'model': 'caller-model', 'messages': [{'role': 'user', 'content': 'local test'}]}
        for selected in (selection('claude-opus-5-5-low'),
                         selection('free/gemini-3.8-flash-high', 'free')):
            a, b = resolve(flags(), selected), resolve(flags(RULES), selected)
            self.assertEqual(a, b)
            self.assertEqual(dedupe_identity(a, payload, {}), dedupe_identity(b, payload, {}))

    def test_kunlun_still_takes_precedence_over_generic_output_tool(self):
        value = flags(RULES)
        value['output_tool'] = {
            'enabled': True, 'match': {'regex': 'gemini'}, 'transports': ['chat'],
            'tool': {'type': 'function', 'function': {'name': 'actual_content',
                'description': 'local fixture', 'parameters': {'type': 'object',
                'properties': {'content': {'type': 'string'}}, 'required': ['content']}}},
            'field': 'content', 'control_prompt': 'local fixture', 'anchors': [],
        }
        target = resolve(value, selection())['policy']['runtime']
        self.assertIn('kunlun_linux', target)
        self.assertNotIn('output_tool', target)
        other = resolve(value, selection('gemini-3.1-pro'))['policy']['runtime']
        self.assertNotIn('kunlun_linux', other)
        self.assertIn('output_tool', other)

    def test_existing_request_snapshot_is_not_mutated_by_new_rule(self):
        value = flags(); selected = selection(); old = resolve(value, selected)
        with policy_scope(old['policy']):
            value['kunlun_linux_model_rules'] = copy.deepcopy(RULES)
            new = resolve(value, selected)
            self.assertNotIn('kunlun_linux', current_policy()['runtime'])
            self.assertIn('kunlun_linux', new['policy']['runtime'])


if __name__ == '__main__':
    unittest.main()
