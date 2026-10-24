"""Numerical checks for research workers, separate from routing policy.

Provenance is the caller's responsibility: only within-job repeats/reloads use
the strict gate. Historical vectors are reported with compare_responses.
"""
import math


def _probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid probability')
    return value


def compare_responses(expected, actual):
    assert expected['usage'] == actual['usage'], 'Usage differs'
    assert set(expected['answers']) == set(actual['answers']), 'Question scope differs'
    assert expected['answers'], 'No answers'
    delta = 0.
    same_choices = True
    for key, old in expected['answers'].items():
        new = actual['answers'][key]
        assert old['type'] == new['type'], 'Question type differs'
        if old['type'] == 'noul':
            delta = max(delta, abs(_probability(old['noul']) - _probability(new['noul'])))
        else:
            assert old['type'] == 'choice', 'Unsupported research control type'
            a, b = old['probabilities'], new['probabilities']
            assert set(a) == set(b) and a, 'Option scope differs'
            for answer, values in [(old, a), (new, b)]:
                assert all(_probability(v) == v for v in values.values())
                assert math.isclose(sum(values.values()), 1, abs_tol=1e-5)
                assert answer['choice'] in values
                assert values[answer['choice']] >= max(values.values()) - 1e-6
            delta = max(delta, max(abs(a[k] - b[k]) for k in a))
            same_choices &= old['choice'] == new['choice']
    return {'maxDelta': delta, 'sameChoices': same_choices}


def require_same_job_parity(expected, actual):
    result = compare_responses(expected, actual)
    assert result['sameChoices'] and result['maxDelta'] <= 1e-5, result
    return result
