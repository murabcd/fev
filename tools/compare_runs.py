"""Summarize saved native typed runs without requerying or tuning a model."""
import argparse
import json
from pathlib import Path

from fev4b.metrics import summarize


def compare(runs):
    indexed = {}
    for name, rows in runs.items():
        values = {(r['id'], r['question']): r for r in rows}
        if len(values) != len(rows):
            raise ValueError('Duplicate question in saved run')
        indexed[name] = values
    keys = next(iter(indexed.values())).keys()
    if any(values.keys() != keys for values in indexed.values()):
        raise ValueError('Runs do not contain the same questions')
    for key in keys:
        labels = {(v[key]['target'], v[key]['type'], v[key]['source'])
                  for v in indexed.values() if v[key]['error'] is None}
        if len(labels) > 1:
            raise ValueError('Runs disagree on target/type/source metadata')
    results = {}
    for name, rows in runs.items():
        subsets = {'all': rows, 'publicTransfer': [r for r in rows if r['source'] != 'authored-russian-typed'],
                   'authoredRussian': [r for r in rows if r['source'] == 'authored-russian-typed']}
        results[name] = {}
        for label, values in subsets.items():
            summary = summarize(values)
            ordinal = [r['ordinalAbsoluteError'] for r in values
                       if r['error'] is None and 'ordinalAbsoluteError' in r]
            summary['ordinalMeanAbsoluteError'] = sum(ordinal)/len(ordinal) if ordinal else None
            results[name][label] = summary
        results[name]['byType'] = {t:summarize([r for r in rows if r['type']==t]) for t in sorted({r['type'] for r in rows})}
        results[name]['bySource'] = {s:summarize([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})}
    paired = {}
    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            av, bv = indexed[a], indexed[b]
            paired[a+' vs '+b] = {
                'aOnlyCorrect': sum(bool(av[k].get('correct')) and not bv[k].get('correct') for k in keys),
                'bOnlyCorrect': sum(bool(bv[k].get('correct')) and not av[k].get('correct') for k in keys),
                'bothCorrect': sum(bool(av[k].get('correct')) and bool(bv[k].get('correct')) for k in keys),
                'neitherCorrect': sum(not av[k].get('correct') and not bv[k].get('correct') for k in keys)}
    return {'models': results, 'pairedOutcomes': paired,
            'interpretation': 'Descriptive comparison on fixed public development and repetitive authored inputs; no significance or unseen customer accuracy claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='NAME=run-directory; repeat for each system')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Output already exists')
    runs = {}
    provenance = {}
    hashes = set()
    for spec in args.run:
        name, directory = spec.split('=', 1)
        if name in runs:
            raise ValueError('Duplicate system name')
        directory = Path(directory)
        report = json.loads((directory/'report.json').read_text())
        p = report['provenance']
        digest = p.get('dataSha256', p.get('datasetSha256'))
        if not digest:
            raise ValueError('Missing input data provenance')
        hashes.add(digest)
        runs[name] = json.loads((directory/'rows.json').read_text())
        provenance[name] = p
    if len(hashes) != 1:
        raise ValueError('Runs use different datasets')
    result = {**compare(runs), 'datasetSha256': hashes.pop(), 'provenance': provenance}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
