"""Text-only typed decisions. No model imports or customer defaults."""
import copy
import math

MODEL_ID = 'fev-4b'


def _json(value, depth=0):
    if depth > 32:
        raise ValueError('JSON nesting exceeds 32 levels')
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json(item, depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json(item, depth + 1)
        return
    raise ValueError('Content must be finite JSON')


def _content(value, *, nullable=False):
    if value is None and nullable:
        return
    if not isinstance(value, (str, dict, list)):
        raise ValueError('Content must be a string, object or array')
    _json(value)


def validate_request(request):
    """Validate without changing instructions, state, keys or option order."""
    if not isinstance(request, dict) or set(request) != {'state', 'questions'}:
        raise ValueError('Expected exactly state and questions; images and assets are unsupported')
    _content(request['state'])
    questions = request['questions']
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
        raise ValueError('Supply 1 to 64 questions')
    for key, question in questions.items():
        if not isinstance(key, str) or not key:
            raise ValueError('Question IDs must be nonempty strings')
        if not isinstance(question, dict) or set(question) - {'type', 'instructions', 'criteria'}:
            raise ValueError('Unsupported question fields')
        kind = question.get('type')
        if kind not in ('choice', 'noul', 'score'):
            raise ValueError('Supported question types: choice, noul, score')
        _content(question.get('instructions'), nullable=True)
        criteria = question.get('criteria')
        if kind == 'choice':
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError('Choice needs 1 to 255 options')
            if any(not isinstance(option, str) or not option for option in criteria):
                raise ValueError('Choice option keys must be nonempty strings')
        elif kind == 'noul':
            if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - {'true', 'false'}):
                raise ValueError('Noul criteria may contain only true and false')
        elif not isinstance(criteria, list) or not 2 <= len(criteria) <= 255:
            raise ValueError('Score needs 2 to 255 ordered levels')
        descriptions = criteria.values() if isinstance(criteria, dict) else (criteria or [])
        for description in descriptions:
            _content(description, nullable=kind != 'score')
    return copy.deepcopy(request)


def _number(value, lower, upper):
    return (type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper)


def validate_response(request, response):
    """Reject incomplete/non-finite native outputs rather than inventing answers."""
    answers = response.get('answers')
    if not isinstance(answers, dict) or set(answers) != set(request['questions']):
        raise RuntimeError('Incomplete typed decision response')
    for key, question in request['questions'].items():
        answer = answers[key]
        kind = question['type']
        if answer.get('type') != kind:
            raise RuntimeError('Native answer type mismatch')
        if kind == 'noul':
            if not _number(answer.get('noul'), 0, 1):
                raise RuntimeError('Invalid Noul probability')
            continue
        keys = list(question['criteria']) if kind == 'choice' else [str(i) for i in range(len(question['criteria']))]
        probabilities = answer.get('probabilities')
        if (not isinstance(probabilities, dict) or set(probabilities) != set(keys)
                or any(not _number(p, 0, 1) for p in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5)
                or not _number(answer.get('confidence'), 0, 1)):
            raise RuntimeError('Invalid typed probability distribution')
        if kind == 'choice':
            if answer.get('choice') not in keys or probabilities[answer['choice']] != max(probabilities.values()):
                raise RuntimeError('Choice does not match distribution')
        else:
            expected_score = sum(int(k) * p for k, p in probabilities.items())
            if (not _number(answer.get('score'), 0, len(keys) - 1)
                    or not math.isclose(answer['score'], expected_score, abs_tol=1e-5)
                    or answer.get('legend') != dict(zip(keys, question['criteria']))):
                raise RuntimeError('Score does not match distribution or legend')
    usage = response.get('usage', {})
    if (type(usage.get('input_tokens')) is not int or usage['input_tokens'] < 1
            or usage.get('output_tokens') != 0 or usage.get('sequences') != 1):
        raise RuntimeError('Typed decisions must use one sequence and generate zero tokens')
    return response


def decide_with_predict(request, *, predict):
    request = validate_request(request)
    response = validate_response(request, predict(request))
    return {'model': MODEL_ID, **response}
