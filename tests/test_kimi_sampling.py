import unittest

from aetherstream.features.kimi_sampling import apply_kimi_sampling_compat


class KimiSamplingTests(unittest.TestCase):
    def test_fixed_sampling_models_omit_only_sampling_fields(self):
        for model in ('transsion/kimi-k3', '[NV]kimi-k3', 'free/kimi-k3',
                      '[G]Kimi-2.7-code', 'kimi-k2.7-code-highspeed', 'kimi-k2.6'):
            with self.subTest(model=model):
                payload = dict(temperature=0.7, top_p=0.98, max_tokens=30000,
                               thinking={'type': 'disabled'}, stream=True)
                self.assertEqual(apply_kimi_sampling_compat(payload, model), ['temperature', 'top_p'])
                self.assertEqual(payload, dict(max_tokens=30000, thinking={'type': 'disabled'}, stream=True))
                self.assertEqual(apply_kimi_sampling_compat(payload, model), [])

    def test_other_models_are_unchanged(self):
        for model in ('gemini-3.1-pro', 'kimi-k2', 'kimi-k2.5', 'kimi-k30', 'not-kimi-k3', None):
            payload = dict(temperature=0.7, top_p=0.98)
            self.assertEqual(apply_kimi_sampling_compat(payload, model), [])
            self.assertEqual(payload, dict(temperature=0.7, top_p=0.98))
