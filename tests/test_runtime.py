"""Small synthetic protocol checks; no model load or network access."""
import copy
import json
import unittest
from pathlib import Path

from fev4b.model import route_with_predict

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((ROOT/'examples/request.json').read_text())

    @staticmethod
    def primary(probabilities):
        return {'answers': {'route': {'type': 'choice', 'probabilities': probabilities,
                 'choice': max(probabilities, key=probabilities.get)}},
                'usage': {'input_tokens': 10, 'output_tokens': 0, 'sequences': 1}}

    def test_full_rules_and_history_reach_applicability(self):
        request = self.payload['request']
        request['state']['history'] = [{'role': 'assistant', 'content': 'Что произошло?'}]
        calls = []
        def predict(actual):
            calls.append(copy.deepcopy(actual))
            if len(calls) == 1:
                self.assertEqual(actual, request)
                return self.primary({'refund': .97, 'delivery': .01, 'unknown': .02})
            self.assertEqual(actual['state']['actualState'], request['state'])
            self.assertEqual(actual['state']['customerRules'], request['questions']['route']['instructions'])
            self.assertEqual(actual['state']['availableRoutes'], request['questions']['route']['criteria'])
            return {'answers': {k: {'type': 'noul', 'noul': .99 if k == 'refund' else .01}
                               for k in actual['questions']},
                    'usage': {'input_tokens': 20, 'output_tokens': 0, 'sequences': 1}}
        result = route_with_predict(request, unknown_key='unknown', predict=predict)
        self.assertEqual(result['decision'], {'kind': 'route', 'key': 'refund'})
        self.assertEqual(result['usage']['input_tokens'], 30)
        self.assertEqual(len(calls), 2)

    def test_unknown_skips_second_call(self):
        calls = []
        def predict(request):
            calls.append(request)
            return self.primary({'refund': .01, 'delivery': .01, 'unknown': .98})
        result = route_with_predict(self.payload['request'], unknown_key='unknown', predict=predict)
        self.assertEqual(result['decision']['kind'], 'clarification')
        self.assertEqual(result['queriedKeys'], [])
        self.assertEqual(result['applicability'], {'refund': None, 'delivery': None})
        self.assertEqual(len(calls), 1)

    def test_invalid_request_never_calls_model(self):
        request = self.payload['request']
        request['state']['inventedHistory'] = 'not actual dialogue'
        def predict(_): raise AssertionError('Must validate first')
        with self.assertRaises(ValueError):
            route_with_predict(request, unknown_key='unknown', predict=predict)

    def test_missing_unknown_never_calls_model(self):
        with self.assertRaises(ValueError):
            route_with_predict(self.payload['request'], unknown_key='absent', predict=lambda _: None)


if __name__ == '__main__': unittest.main()
