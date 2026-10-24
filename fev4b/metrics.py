"""Distribution metrics, with all failed questions retained in the denominator."""
import math

from .data import option_keys, target_index


def distribution(question, answer):
    if answer.get('type') != question['type']:
        raise ValueError('Answer type mismatch')
    keys = option_keys(question)
    if question['type'] == 'noul':
        p = answer['noul']
        values = [1-p, p]
    else:
        if set(answer['probabilities']) != set(keys):
            raise ValueError('Wrong probability keys')
        values = [answer['probabilities'][k] for k in keys]
    if (any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values)
            or not math.isclose(sum(values), 1, abs_tol=.02)):
        raise ValueError('Invalid distribution')
    # Hosted APIs serialize rounded probabilities. Normalize only within the
    # API tolerance; keep the original response in predictions.jsonl.
    return [p/sum(values) for p in values]


def rows_for(record, response=None, error=None):
    rows = []
    if response is not None and set(response['answers']) != set(record['questions']):
        raise ValueError('Incomplete response')
    for key, question in record['questions'].items():
        row = {'id': record['id'], 'question': key, 'type': question['type'],
               'source': record.get('source', record.get('domain', 'unspecified')), 'error': error}
        if error is None:
            values = distribution(question, response['answers'][key])
            target = target_index(question)
            predicted = max(range(len(values)), key=values.__getitem__)
            row.update(correct=predicted == target, confidence=max(values),
                       probabilities=values, target=target,
                       brier=sum((p - int(i == target))**2 for i, p in enumerate(values)),
                       nll=-math.log(max(values[target], 1e-12)))
            if question['type'] == 'score':
                row['ordinalAbsoluteError'] = abs(sum(i*p for i, p in enumerate(values)) - target)
        rows.append(row)
    return rows


def summarize(rows):
    good = [r for r in rows if r['error'] is None]
    bins = [[] for _ in range(10)]
    for row in good:
        bins[min(9, int(row['confidence']*10))].append(row)
    ece = sum(len(b) * abs(sum(r['correct'] for r in b)/len(b) - sum(r['confidence'] for r in b)/len(b))
              for b in bins if b) / len(good) if good else None
    coverage = []
    for threshold in (.5, .7, .8, .9, .95, .99):
        accepted = [r for r in good if r['confidence'] >= threshold]
        coverage.append({'threshold': threshold, 'accepted': len(accepted),
                         'coverage': len(accepted)/len(rows) if rows else 0,
                         'errorRate': sum(not r['correct'] for r in accepted)/len(accepted) if accepted else None})
    return {'questions': len(rows), 'answered': len(good), 'errors': len(rows)-len(good),
            'correct': sum(r['correct'] for r in good),
            'accuracyAllQuestions': sum(r['correct'] for r in good)/len(rows) if rows else None,
            'brierAnswered': sum(r['brier'] for r in good)/len(good) if good else None,
            'nllAnswered': sum(r['nll'] for r in good)/len(good) if good else None,
            'ece10Answered': ece,
            'confidentWrongAt09': sum(not r['correct'] and r['confidence'] >= .9 for r in good),
            'coverageByFixedThreshold': coverage,
            'coverageInterpretation': 'Observed on this dataset; not a future error guarantee. Thresholds not tuned here.'}


def report_for(rows):
    return {'overall': summarize(rows),
            'byType': {k: summarize([r for r in rows if r['type'] == k]) for k in sorted({r['type'] for r in rows})},
            'bySource': {k: summarize([r for r in rows if r['source'] == k]) for k in sorted({r['source'] for r in rows})}}
