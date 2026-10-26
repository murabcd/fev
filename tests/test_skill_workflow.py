"""Exercise the distributed skill helpers without a GPU, network or model weights."""
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fev4b.benchmark import evaluate
from fev4b.data import load_records
from fev4b.integrity import file_sha256
from tools.train_hf import run_experiment

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'skills/fev-finetune/scripts'
sys.path.insert(0, str(SCRIPTS))
# compare.py is distributed with the skill, not imported by serving code.
spec = importlib.util.spec_from_file_location('skill_compare', SCRIPTS/'compare.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def records(count=40):
    return [{'id': str(i), 'group': f'family-{i}', 'domain': ('library', 'travel', 'retail')[i % 3],
             'state': f'Request {i}', 'questions': {
                 'q': {'type': 'choice', 'criteria': {'z': 'First', 'a': 'Second'}, 'label': 'z'}}}
            for i in range(count)]


def write_data(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def cli(name, *args, success=True):
    result = subprocess.run([sys.executable, '-B', str(SCRIPTS/name), *map(str, args)],
                            capture_output=True, text=True)
    if (result.returncode == 0) != success:
        raise AssertionError(result.stdout + result.stderr)
    return result


class SkillWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root/'input.jsonl'
        write_data(self.input, records())

    def prepare(self, *args, success=True):
        out = self.root/'split'
        cli('prepare_data.py', '--checkout', ROOT, '--data', self.input, '--out', out,
            *args, success=success)
        return out

    def saved_run(self, name, rows, predictor):
        out = self.root/name
        evaluate(rows, predictor, out, provenance={
            'dataSha256': file_sha256(self.input), 'modelManifestSha256': 'frozen-manifest',
            'execution': 'reference', 'requestedModel': 'fev-4b', 'adaptation': None})
        return out

    @staticmethod
    def correct(_):
        return {'answers': {'q': {'type': 'choice', 'probabilities': {'z': .95, 'a': .05}}}}

    def test_splits_preserve_option_order_and_holdout_domain(self):
        out = self.prepare('--holdout-domain', 'travel')
        parts = {p.stem: load_records(p) for p in out.glob('*.jsonl')}
        self.assertTrue(all(list(r['questions']['q']['criteria']) == ['z', 'a']
                            for rows in parts.values() for r in rows))
        self.assertTrue(all(r['domain'] != 'travel' for key in ('train', 'development', 'calibration')
                            for r in parts[key]))
        self.assertEqual(sum(r['domain'] == 'travel' for r in parts['test']), 13)
        manifest = json.loads((out/'manifest.json').read_text())
        self.assertEqual(manifest['files']['test.jsonl']['sha256'], file_sha256(out/'test.jsonl'))
        self.prepare(success=False)

    def test_reject_duplicate_ids_and_missing_families_without_output(self):
        for bad in ([records()[0], records()[0]], [{k: v for k, v in r.items() if k != 'group'} for r in records()]):
            write_data(self.input, bad)
            self.prepare(success=False)
            self.assertFalse((self.root/'split').exists())

    def test_csv_typed_mapping_and_strict_boolean(self):
        template = json.loads((ROOT/'skills/fev-finetune/assets/questions.example.json').read_text())
        spec_path, csv_path = self.root/'questions.json', self.root/'input.csv'
        spec_path.write_text(json.dumps(template))
        def write_csv(boolean):
            with csv_path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'domain', 'group', 'text', 'c', 'n', 's'])
                writer.writeheader()
                writer.writerows({'id': str(i), 'domain': r['domain'], 'group': r['group'],
                                  'text': r['state'], 'c': 'cancel', 'n': boolean, 's': 0}
                                 for i, r in enumerate(records()))
        options = ['--checkout', ROOT, '--data', csv_path, '--spec', spec_path,
                   '--label', 'action=c', '--label', 'cancel_requested=n', '--label', 'urgency=s']
        write_csv('yes')
        cli('prepare_data.py', *options, '--out', self.root/'bad', success=False)
        self.assertFalse((self.root/'bad').exists())
        write_csv('true')
        out = self.root/'converted'
        cli('prepare_data.py', *options, '--out', out)
        row = load_records(out/'train.jsonl')[0]
        self.assertIs(row['questions']['cancel_requested']['label'], True)
        self.assertEqual(row['questions']['urgency']['label'], 0)

    def test_minimal_job_package_and_preflight_leak_rejection(self):
        split = self.prepare()
        out = self.root/'package'
        cli('package_job.py', '--checkout', ROOT, '--split', split, '--out', out)
        manifest = json.loads((out/'package.json').read_text())
        self.assertFalse(manifest['computeAllocated'])
        self.assertEqual(set(manifest['files']), {'source.zip', 'train_hf.py', 'common.py', 'compare.py',
                         'train.jsonl', 'development.jsonl', 'calibration.jsonl', 'test.jsonl'})
        self.assertTrue(all(file_sha256(out/name) == digest for name, digest in manifest['files'].items()))
        with zipfile.ZipFile(out/'source.zip') as archive:
            self.assertTrue(all(n.startswith(('fev/', 'fev4b/')) and n.endswith('.py') for n in archive.namelist()))
            self.assertIn('fev4b/train.py', archive.namelist())
        (split/'test.jsonl').write_bytes((split/'train.jsonl').read_bytes())
        cli('package_job.py', '--checkout', ROOT, '--split', split, '--out', self.root/'bad', success=False)
        self.assertFalse((self.root/'bad').exists())

    def test_paired_report_counts_failures_and_domains(self):
        rows = records(6)
        write_data(self.input, rows)
        a = self.saved_run('a', rows, self.correct)
        def predict(request):
            if request['state'] == 'Request 0':
                raise RuntimeError('do not expose this text')
            return self.correct(request)
        b = self.saved_run('b', rows, predict)
        result, errors = comparison.build_report(ROOT, self.input, a, b, iterations=200)
        self.assertEqual(result['candidate']['overall']['questions'], 6)
        self.assertEqual(result['candidate']['overall']['errors'], 1)
        self.assertEqual(result['paired']['baselineOnlyCorrect'], 1)
        self.assertEqual(result['domainRegressions'], ['library'])
        self.assertEqual(len(errors), 1)
        self.assertEqual(result['paired']['independentClusters'], 6)
        self.assertFalse(result['automaticAdoption'])
        out = self.root/'comparison'
        cli('compare.py', '--checkout', ROOT, '--data', self.input, '--baseline', a, '--candidate', b,
            '--out', out, '--hourly-rate-usd', 1.2, '--billed-seconds', 600, '--rate-source', 'fixture quote')
        self.assertAlmostEqual(json.loads((out/'comparison.json').read_text())['computeEstimate']['usd'], .2)
        self.assertTrue((out/'summary.md').is_file())
        cli('compare.py', '--checkout', ROOT, '--data', self.input, '--baseline', a, '--candidate', b,
            '--out', self.root/'bad-cost', '--hourly-rate-usd', 'nan', '--billed-seconds', 1,
            '--rate-source', 'fixture', success=False)

    def test_pairing_rejects_changed_hash_duplicate_predictions_and_consumed_test(self):
        rows = records(4)
        write_data(self.input, rows)
        a = self.saved_run('a', rows, self.correct)
        b = self.saved_run('b', rows, self.correct)
        original = (b/'predictions.jsonl').read_text()
        (b/'predictions.jsonl').write_text(original + original.splitlines()[0] + '\n')
        with self.assertRaises(ValueError):
            comparison.build_report(ROOT, self.input, a, b)
        (b/'predictions.jsonl').write_text(original)
        report = json.loads((b/'report.json').read_text())
        report['provenance']['adaptation'] = {'consumed': {'ids': ['0']}}
        (b/'report.json').write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            comparison.build_report(ROOT, self.input, a, b)
        report['provenance']['adaptation'] = None
        report['provenance']['dataSha256'] = 'changed'
        (b/'report.json').write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            comparison.build_report(ROOT, self.input, a, b)

    def test_worker_rejects_tampered_input_before_importing_gpu_dependencies(self):
        split = self.prepare()
        inputs = self.root/'inputs'
        cli('package_job.py', '--checkout', ROOT, '--split', split, '--out', inputs)
        with (inputs/'test.jsonl').open('a') as stream:
            stream.write('\n')
        result = subprocess.run([sys.executable, str(inputs/'train_hf.py'), '--inputs', str(inputs),
            '--output', str(self.root/'job'), '--revision', 'a' * 40], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Packaged input hash mismatch: test.jsonl', result.stderr)
        self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_cluster_interval_does_not_treat_questions_or_variants_as_independent(self):
        rows = records(6)
        for row in rows:
            row['group'] = 'one-family'
        a = [{'id': r['id'], 'question': 'q', 'correct': False} for r in rows]
        b = [{**row, 'correct': True} for row in a]
        from fev4b import data
        result = comparison.paired_summary(rows, a, b, data, iterations=200)
        self.assertEqual(result['independentClusters'], 1)
        self.assertEqual(result['accuracyDelta'], 1)
        self.assertIsNone(result['clusterBootstrap95'])
        rows = records(4)
        rows[1]['group'] = rows[0]['group']
        rows[2]['state'] = rows[1]['state']
        assigned = comparison.clusters(rows, data)
        self.assertEqual(len(set(assigned.values())), 2)

    def test_bootstrap_installed_helpers_work_without_repository_siblings(self):
        source = self.root/'source'
        source.mkdir()
        subprocess.run(['git', 'init', '-q', str(source)], check=True)
        (source/'fev4b').mkdir()
        (source/'fev4b/train.py').write_text('# fixture\n')
        subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(source), '-c', 'user.name=Fixture', '-c',
                        'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        installed = self.root/'installed'
        shutil.copytree(ROOT/'skills/fev-finetune', installed)
        result = subprocess.run([sys.executable, str(installed/'scripts/bootstrap.py'), '--source', str(source),
                                 '--revision', revision, '--out', str(self.root/'checkout')], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root/'checkout/fev4b/train.py').is_file())
        cli('bootstrap.py', '--source', source, '--revision', 'main', '--out', self.root/'bad', success=False)
        self.assertFalse((self.root/'bad').exists())

    def test_worker_runs_all_variants_on_same_test_and_reports_even_prediction_failures(self):
        output = self.root/'run'
        output.mkdir()
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            if 'fev4b.benchmark' in command:
                out = Path(command[command.index('--out') + 1])
                out.mkdir()
                failures = int(out.name == 'raw-test')
                (out/'report.json').write_text(json.dumps({'overall': {'errors': failures}}))
                return subprocess.CompletedProcess(command, failures)
            return subprocess.CompletedProcess(command, 0)
        with patch('tools.train_hf.subprocess.run', side_effect=fake_run):
            with self.assertRaises(RuntimeError):
                run_experiment(ROOT, self.root/'model', self.root/'inputs', output,
                               steps=6, learning_rate=0.00002, seed=42)
        runs = [c for c in calls if 'fev4b.benchmark' in c]
        self.assertEqual(len(runs), 3)
        self.assertEqual({c[c.index('--data') + 1] for c in runs}, {str(self.root/'inputs/test.jsonl')})
        self.assertNotIn('--adaptation-dir', runs[0])
        self.assertEqual(sum(str(self.root/'inputs/compare.py') in c for c in calls), 2)
        finished = json.loads((output/'complete.json').read_text())
        self.assertEqual(finished['predictionErrors']['raw-test'], 1)
        self.assertFalse(finished['adopted'])


if __name__ == '__main__':
    unittest.main()
