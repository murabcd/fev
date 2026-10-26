"""Re-score paired predictions and estimate uncertainty by document/family cluster."""
import argparse
import json
import math
import random
from pathlib import Path

from common import load_core, sha256, write_json


def score_run(directory, records, data_hash, metrics):
    report = json.loads((directory / 'report.json').read_text())
    if report.get('provenance', {}).get('dataSha256') != data_hash:
        raise ValueError('Run does not identify the exact frozen test file')
    saved = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines() if line.strip()]
    indexed = {row['id']: row for row in saved}
    if len(indexed) != len(saved) or set(indexed) != {r['id'] for r in records}:
        raise ValueError('Prediction IDs are duplicate, missing or unexpected')
    rows = []
    for record in records:
        item = indexed[record['id']]
        if ('response' in item) == ('error' in item):
            raise ValueError('Each prediction must contain exactly one response or error')
        if 'error' in item and (not isinstance(item['error'], str) or not item['error']):
            raise ValueError('A failed prediction requires a nonempty error name')
        scored = metrics.rows_for(record, item.get('response'), item.get('error'))
        for row in scored:
            row['domain'] = record['domain']
        rows.extend(scored)
    return rows, report['provenance']


def clusters(records, data):
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, record in enumerate(records):
        for key in (('state', data.state_hash(record)), ('group', record['group'])):
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    return {record['id']: find(i) for i, record in enumerate(records)}


def paired_summary(records, baseline, candidate, data, iterations=2000, seed=42):
    assignment = clusters(records, data)
    groups, counts = {}, {'bothCorrect': 0, 'baselineOnlyCorrect': 0,
                          'candidateOnlyCorrect': 0, 'neitherCorrect': 0}
    for a, b in zip(baseline, candidate):
        if (a['id'], a['question']) != (b['id'], b['question']):
            raise ValueError('Paired question order differs')
        ac, bc = bool(a.get('correct')), bool(b.get('correct'))
        key = ('bothCorrect' if ac and bc else 'baselineOnlyCorrect' if ac else
               'candidateOnlyCorrect' if bc else 'neitherCorrect')
        counts[key] += 1
        groups.setdefault(assignment[a['id']], []).append(int(bc) - int(ac))
    totals = [(sum(values), len(values)) for values in groups.values()]
    interval = None
    if len(totals) >= 2:
        rng = random.Random(seed)
        samples = []
        for _ in range(iterations):
            draw = rng.choices(totals, k=len(totals))
            samples.append(sum(d for d, n in draw) / sum(n for d, n in draw))
        samples.sort()
        interval = [samples[int(.025 * iterations)], samples[min(iterations - 1, int(.975 * iterations))]]
    return {**counts, 'accuracyDelta': (counts['candidateOnlyCorrect'] - counts['baselineOnlyCorrect']) / len(baseline),
            'clusterBootstrap95': interval, 'independentClusters': len(totals),
            'iterations': iterations, 'seed': seed,
            'interpretation': 'Descriptive paired interval on supplied families; not proof of domain transfer. '
                              'Few clusters or unmarked semantic duplicates weaken inference.'}


def build_report(checkout, test, baseline, candidate, *, iterations=2000, seed=42):
    data, metrics = load_core(checkout)
    records = data.load_records(test)
    if any(not r.get('domain') or not r.get('group') for r in records):
        raise ValueError('Comparison requires domain and family group metadata')
    a, ap = score_run(baseline, records, sha256(test), metrics)
    b, bp = score_run(candidate, records, sha256(test), metrics)
    for field in ('modelManifestSha256', 'execution', 'requestedModel'):
        if not ap.get(field) or ap[field] != bp.get(field):
            raise ValueError(f'Paired runs must share recorded {field}')
    if ap.get('adaptation') is not None:
        raise ValueError('Baseline must be the unchanged packaged model')
    consumed = (bp.get('adaptation') or {}).get('consumed', {})
    if ({r['id'] for r in records} & set(consumed.get('ids', [])) or
            {data.state_hash(r) for r in records} & set(consumed.get('states', [])) or
            {r['group'] for r in records} & set(consumed.get('groups', []))):
        raise ValueError('Final test overlaps adapter training, development or calibration')
    def summary(rows):
        ordinal = [r['ordinalAbsoluteError'] for r in rows if 'ordinalAbsoluteError' in r]
        return {**metrics.report_for(rows),
            'ordinalAbsoluteErrorAnswered': sum(ordinal)/len(ordinal) if ordinal else None, 'byDomain': {
            domain: metrics.summarize([r for r in rows if r['domain'] == domain])
            for domain in sorted({r['domain'] for r in rows})}}
    result = {'dataSha256': sha256(test), 'baseline': summary(a), 'candidate': summary(b),
              'provenance': {'baseline': ap, 'candidate': bp},
              'paired': paired_summary(records, a, b, data, iterations, seed),
              'decision': 'review_required', 'automaticAdoption': False}
    result['domainRegressions'] = [domain for domain, value in result['baseline']['byDomain'].items()
        if result['candidate']['byDomain'][domain]['accuracyAllQuestions'] < value['accuracyAllQuestions']]
    errors = [{'id': row['id'], 'question': row['question'], 'domain': row['domain'],
               'baselineCorrect': bool(old.get('correct')), 'candidate': row}
              for old, row in zip(a, b) if row.get('error') or not row.get('correct')]
    return result, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hourly-rate-usd', type=float)
    parser.add_argument('--billed-seconds', type=float)
    parser.add_argument('--rate-source')
    args = parser.parse_args()
    if args.iterations < 100:
        raise ValueError('Use at least 100 bootstrap draws')
    if args.out.exists():
        raise ValueError('Report output already exists')
    cost = (args.hourly_rate_usd, args.billed_seconds, args.rate_source)
    if any(v is not None for v in cost) and (any(v is None for v in cost) or
            not args.rate_source.strip() or any(not math.isfinite(v) or v < 0 for v in cost[:2])):
        raise ValueError('Cost requires a finite nonnegative hourly rate, billed seconds and a rate source')
    result, errors = build_report(args.checkout, args.data, args.baseline, args.candidate,
                                  iterations=args.iterations, seed=args.seed)
    if args.rate_source is not None:
        result['computeEstimate'] = {'usd': args.hourly_rate_usd * args.billed_seconds / 3600,
            'hourlyRateUsd': args.hourly_rate_usd, 'billedSeconds': args.billed_seconds,
            'rateSource': args.rate_source, 'meaning': 'Compute estimate only; not an invoice or monthly serving cost'}
    args.out.mkdir(parents=True)
    write_json(args.out / 'comparison.json', result)
    (args.out / 'candidate-errors.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in errors))
    a, b = result['baseline']['overall'], result['candidate']['overall']
    (args.out / 'summary.md').write_text(
        '# Paired Fev comparison\n\n| Metric | Baseline | Candidate |\n|---|---:|---:|\n' +
        ''.join(f'| {key} | {a[key]} | {b[key]} |\n' for key in
                ('questions', 'accuracyAllQuestions', 'errors', 'brierAnswered', 'nllAnswered', 'confidentWrongAt09')) +
        f'\nAccuracy delta: {result["paired"]["accuracyDelta"]:+.4f}; family-cluster 95% interval: '
        f'{result["paired"]["clusterBootstrap95"]}.\n\n'
        'Review domain regressions, confident errors and the predeclared acceptance rules. '
        'No automatic adoption. Calibration and weight changes are separate interventions; '
        'compare the raw adapter as well. These scores do not establish serving capacity.\n')
    print(json.dumps(result['paired'], indent=2))


if __name__ == '__main__':
    main()
