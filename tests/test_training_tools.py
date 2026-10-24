"""Leakage, label and measurement checks independent of CUDA."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from fev4b.data import split_records, require_disjoint, target_index
from fev4b.metrics import rows_for, summarize
from fev4b.calibrate import fit_temperature, rescale
from fev4b.benchmark import evaluate
from fev4b.adaptation import base_identity, verify_adaptation
from fev4b.integrity import file_sha256
from tools.compare_runs import compare


def record(i, group=None):
    return {'id': str(i), 'group': group or str(i), 'state': str(i), 'questions': {
        'q': {'type': 'choice', 'criteria': {'yes': None, 'no': None}, 'label': 'yes'}}}


class TrainingToolsTests(unittest.TestCase):
    def test_overlay_rejects_other_base_or_modified_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'adapter').mkdir()
            config = {'r': 8, 'lora_alpha': 16}
            (root/'adapter/adapter_config.json').write_text(json.dumps(config))
            manifest = {'files': {'checkpoint/model.safetensors': {'bytes': 4, 'sha256': 'base'},
                                  'adapter/adapter_model.safetensors': {'bytes': 4, 'sha256': 'shared'}}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            overlay = root/'overlay'
            overlay.mkdir()
            (overlay/'adapter_config.json').write_text(json.dumps(config))
            (overlay/'adapter_model.safetensors').write_bytes(b'test')
            meta = {'format': 'fev-adaptation-v1', 'baseIdentity': base_identity(root), 'temperature': 1.5,
                    'files': {name: file_sha256(overlay/name) for name in
                              ('adapter_config.json', 'adapter_model.safetensors')}}
            (overlay/'adaptation.json').write_text(json.dumps(meta))
            self.assertEqual(verify_adaptation(root, overlay)['temperature'], 1.5)
            (overlay/'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_adaptation(root, overlay)
            (overlay/'adapter_model.safetensors').write_bytes(b'test')
            manifest['files']['checkpoint/model.safetensors']['sha256'] = 'other'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                verify_adaptation(root, overlay)

    def test_comparison_rejects_different_labels_and_preserves_paired_counts(self):
        a = rows_for(record(0), {'answers': {'q': {'type': 'choice', 'probabilities': {'yes': .9, 'no': .1}}}})
        b = rows_for(record(0), {'answers': {'q': {'type': 'choice', 'probabilities': {'yes': .1, 'no': .9}}}})
        result = compare({'a': a, 'b': b})
        self.assertEqual(result['pairedOutcomes']['a vs b']['aOnlyCorrect'], 1)
        b[0]['target'] = 1
        with self.assertRaises(ValueError):
            compare({'a': a, 'b': b})

    def test_group_bridge_does_not_leak(self):
        records = [record(i) for i in range(30)]
        duplicate = record(31, group='0')
        duplicate['state'] = '1'
        records.append(duplicate)
        split = split_records(records)
        require_disjoint(*split.values())
        partition = next(rows for rows in split.values() if any(r['id'] == '0' for r in rows))
        self.assertTrue({'0', '1', '31'} <= {r['id'] for r in partition})
        with self.assertRaises(ValueError):
            require_disjoint([record(0)], [record(1, group='0')])

    def test_label_types_are_strict(self):
        with self.assertRaises(ValueError):
            target_index({'type': 'noul', 'label': 1})
        with self.assertRaises(ValueError):
            target_index({'type': 'score', 'criteria': ['low', 'high'], 'label': True})

    def test_errors_remain_in_denominator_and_confidence_is_top_probability(self):
        response = {'answers': {'q': {'type': 'choice', 'probabilities': {'yes': .9, 'no': .1}}}}
        rows = rows_for(record(0), response) + rows_for(record(1), error='Timeout')
        report = summarize(rows)
        self.assertEqual(report['accuracyAllQuestions'], .5)
        self.assertEqual(report['coverageByFixedThreshold'][3]['coverage'], .5)
        response['answers']['q']['probabilities'] = {'yes': 2., 'no': -1.}
        with self.assertRaises(ValueError):
            rows_for(record(0), response)

    def test_temperature_does_not_change_argmax(self):
        values = [.05, .15, .8]
        for factor in (.2, 1., 5.):
            self.assertEqual(max(range(3), key=rescale(values, factor).__getitem__), 2)
        rows = [{'error': None, 'probabilities': [.99, .01], 'target': 1} for _ in range(3)]
        _, report = fit_temperature(rows)
        self.assertLessEqual(report['afterNll'], report['beforeNll'])

    def test_benchmark_keeps_failed_records(self):
        def predict(request):
            if request['state'] == '1':
                raise RuntimeError('failed')
            return {'model': 'fake', 'answers': {'q': {'type': 'choice', 'probabilities': {'yes': .8, 'no': .2}}}}
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate([record(0), record(1)], predict, Path(directory)/'run', provenance={})
            self.assertEqual(report['overall']['questions'], 2)
            self.assertEqual(report['overall']['errors'], 1)


if __name__ == '__main__':
    unittest.main()
