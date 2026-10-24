"""Contract checks with a fake predictor, plus real TypeSafe SDK over HTTP.

These tests validate the interface, not learned model quality.
"""
import copy
import json
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fev4b.decisions import decide_with_predict, validate_request
from fev4b.server import MAX_BODY_BYTES, create_server

ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __init__(self):
        self.calls = []

    def decide(self, request):
        self.calls.append(copy.deepcopy(request))
        def predict(actual):
            answers = {}
            for key, question in actual['questions'].items():
                kind = question['type']
                if kind == 'noul':
                    answers[key] = {'type': kind, 'noul': .9}
                    continue
                keys = list(question['criteria']) if kind == 'choice' else [str(i) for i in range(len(question['criteria']))]
                probabilities = dict.fromkeys(keys, 0.)
                probabilities[keys[0]] = 1.
                answers[key] = {'type': kind, 'probabilities': probabilities, 'confidence': 1.}
                if kind == 'choice':
                    answers[key]['choice'] = keys[0]
                else:
                    answers[key].update(score=0., legend=dict(zip(keys, question['criteria'])))
            return {'answers': answers, 'usage': {'input_tokens': 100, 'output_tokens': 0, 'sequences': 1}}
        return decide_with_predict(request, predict=predict)


class DecisionTests(unittest.TestCase):
    def test_reject_bad_schema_before_inference(self):
        examples = [
            {'state': 'x', 'questions': {}},
            {'state': 'x', 'questions': {'q': {'type': 'box'}}},
            {'state': 'x', 'questions': {'q': {'type': 'score', 'criteria': ['one']}}},
            {'state': 'x', 'questions': {'q': {'type': 'noul', 'criteria': {'other': 'x'}}}},
            {'state': {'x': float('nan')}, 'questions': {'q': {'type': 'noul'}}},
            {'state': 'x', 'questions': {'q': {'type': 'choice', 'criteria': {'': None}}}},
        ]
        for request in examples:
            with self.subTest(request=request), self.assertRaises(ValueError):
                decide_with_predict(request, predict=lambda _: self.fail('Called model for invalid input'))

    def test_deep_copy_and_mixed_answer_validation(self):
        payload = json.loads((ROOT/'examples/typed-request.json').read_text())
        request = {k: payload[k] for k in ('state', 'questions')}
        original = copy.deepcopy(request)
        result = FakeModel().decide(request)
        self.assertEqual(request, original)
        self.assertEqual(result['model'], 'fev-4b')
        self.assertEqual(result['answers']['topic']['choice'], 'billing')
        self.assertEqual(result['answers']['urgency']['legend']['0'], request['questions']['urgency']['criteria'][0])

    def test_invalid_native_answer_is_not_returned(self):
        request = {'state': 'x', 'questions': {'q': {'type': 'noul'}}}
        with self.assertRaises(RuntimeError):
            decide_with_predict(request, predict=lambda _: {
                'answers': {'q': {'type': 'noul', 'noul': float('nan')}},
                'usage': {'input_tokens': 10, 'output_tokens': 0, 'sequences': 1}})


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.server = create_server(self.model, ('127.0.0.1', 0), api_key='local-test')
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def post(self, body, *, key='local-test', **headers):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        request = Request(self.url+'/v1/systemone', data=raw, headers={
            'Authorization': 'Bearer '+key, 'Content-Type': 'application/json', **headers})
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def test_real_sdk_plain_and_compressed_mixed_questions(self):
        from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
        for compression in (False, 'zstd'):
            # Exceed SDK's automatic compression size threshold.
            state = {'message': 'Оплата списана дважды. ' * 200, 'history': []}
            with TypeSafeClient(api_key='local-test', base_url=self.url,
                               model='fev-4b', compression=compression) as client:
                response = client.system_one(state=state, questions={
                    'team': Choice(criteria={'billing': 'Платежи', 'delivery': 'Доставка'}),
                    'refund': Noul(instructions='Нужен возврат?'),
                    'urgency': Score(criteria=['Низкая', 'Высокая']),
                })
                self.assertEqual(response.choices['team'].choice, 'billing')
                self.assertEqual(response.nouls['refund'].noul, .9)
                self.assertEqual(response.scores['urgency'].score, 0.)
                self.assertEqual(response.usage.output_tokens, 0)
                self.assertTrue(response.request_id)
                self.assertEqual(client.models.list().models[0].name, 'fev-4b')
                self.assertEqual(self.model.calls[-1]['state'], state)
        self.assertEqual(len(self.model.calls), 2)

    def test_auth_unknown_models_assets_and_duplicate_keys(self):
        payload = json.loads((ROOT/'examples/typed-request.json').read_text())
        self.assertEqual(self.post(payload, key='wrong')[0], 401)
        payload['model'] = 'unavailable'
        self.assertEqual(self.post(payload)[0], 422)
        payload['model'] = 'fev-4b'
        payload['assets'] = {'photo': {'url': 'http://example.invalid'}}
        self.assertEqual(self.post(payload)[0], 422)
        self.assertEqual(self.post(b'{"model":"fev-4b","model":"other"}')[0], 422)
        self.assertEqual(self.model.calls, [])

    def test_bounded_decompression(self):
        import zstandard
        large = zstandard.ZstdCompressor().compress(b' ' * (MAX_BODY_BYTES + 100))
        self.assertIn(self.post(large, **{'Content-Encoding': 'zstd'})[0], (400, 413))
        self.assertEqual(self.model.calls, [])


if __name__ == '__main__':
    unittest.main()
