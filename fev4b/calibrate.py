"""Fit one temperature on a separate calibration partition, writing a new overlay."""
import argparse
import json
import math
import shutil
from pathlib import Path

from .adaptation import verify_adaptation
from .data import load_records, request_of, state_hash
from .metrics import rows_for, report_for


def rescale(values, factor):
    logits = [math.log(max(p, 1e-12))/factor for p in values]
    maximum = max(logits)
    exps = [math.exp(v-maximum) for v in logits]
    return [v/sum(exps) for v in exps]


def fit_temperature(rows):
    if not rows or any(r['error'] is not None for r in rows):
        raise ValueError('Calibration needs complete valid predictions')
    def nll(factor):
        return sum(-math.log(max(rescale(r['probabilities'], factor)[r['target']], 1e-12)) for r in rows)/len(rows)
    # Include the unchanged temperature; minimize calibration NLL only.
    factors = sorted({1., *(math.exp(-2 + i*.02) for i in range(201))})
    best = min(factors, key=nll)
    return best, {'beforeNll': nll(1.), 'afterNll': nll(best),
                  'factorBounds': [math.exp(-2), math.exp(2)],
                  'interpretation': 'In-sample calibration fit; independent test needed before adoption'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--adaptation-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    meta = verify_adaptation(args.model_dir, args.adaptation_dir)
    records = load_records(args.data)
    consumed = meta['consumed']
    if ({r['id'] for r in records} & set(consumed['ids'])
            or {state_hash(r) for r in records} & set(consumed['states'])
            or {r['group'] for r in records if 'group' in r} & set(consumed['groups'])):
        raise ValueError('Calibration overlaps training or development')
    if args.out.exists():
        raise ValueError('Calibration output already exists')
    from .model import Fev4B
    model = Fev4B(args.model_dir, adaptation_dir=args.adaptation_dir)
    rows = [r for record in records for r in rows_for(record, model.decide(request_of(record)))]
    factor, report = fit_temperature(rows)
    args.out.mkdir(parents=True)
    for name in meta['files']:
        shutil.copyfile(args.adaptation_dir/name, args.out/name)
    meta['temperature'] *= factor
    meta['temperatureStatus'] = 'fitted on separate calibration data; test not read'
    meta['consumed']['ids'] += [r['id'] for r in records]
    meta['consumed']['states'] += [state_hash(r) for r in records]
    meta['consumed']['groups'] += [r['group'] for r in records if 'group' in r]
    (args.out/'adaptation.json').write_text(json.dumps(meta, indent=2)+'\n')
    (args.out/'calibration-report.json').write_text(json.dumps({**report, 'factor': factor,
        'calibrationMetricsBefore': report_for(rows), 'records': len(records)}, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
