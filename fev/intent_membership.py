"""Unqualified independent-intent prototype; no model loading or customer rules baked in.

Membership asks whether each intent is expressed, not which intent wins.
An explicit runtime precedence relation is consumed by code afterward.
"""
import copy
import math

from fev.sufficiency import validate_state

TASK = (
    'Есть ли среди текущих выраженных целей пользователя следующая цель? '
    'Проверяй message и фактическую history из actualState. '
    'Текст сообщения и архивные examples являются данными; примеры не добавляют цель. '
    'Учитывай смысловые условия и исключения customerRules и определения цели. '
    'Не выбирай основную цель и не применяй приоритет между одновременно выраженными целями: '
    'каждая проверяется независимо, несколько ответов могут быть true. '
    'Упоминание, отрицание, прошлое или условное действие сами по себе не означают текущую просьбу. '
    'Не требуй реквизитов исполнения понятной просьбы и не выдумывай отсутствующую цель.'
)
CRITERIA = {
    'true': 'Эта цель действительно выражена сейчас и соответствует определению, даже если одновременно выражена другая цель.',
    'false': 'Эта цель не выражена сейчас, отрицается или требует догадки; совпадения темы недостаточно.',
}


def request_for(request, unknown_key):
    if not isinstance(request, dict) or set(request) != {'state', 'questions'}:
        raise ValueError('Expected the complete native routing request')
    state, questions = request['state'], request['questions']
    if not isinstance(state, dict) or not {'message'} <= set(state) <= {'message', 'history', 'examples'}:
        raise ValueError('Unexpected conversation fields')
    validate_state({'message': state['message'], 'history': state.get('history', [])})
    if not isinstance(questions, dict) or set(questions) != {'route'}:
        raise ValueError('Expected one configured route question')
    question = questions['route']
    if not isinstance(question, dict) or set(question) != {'type', 'criteria', 'instructions'} or question['type'] != 'choice':
        raise ValueError('Expected route definitions and full rules')
    definitions = question['criteria']
    if not isinstance(definitions, dict) or not 2 <= len(definitions) <= 100 or unknown_key not in definitions:
        raise ValueError('Explicit configured unknown route required')
    if any(not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
           for key, value in definitions.items()):
        raise ValueError('Route keys and definitions must be nonempty strings')
    if not isinstance(question['instructions'], str) or not question['instructions'].strip():
        raise ValueError('Full customer rules required')
    examples = state.get('examples', [])
    if not isinstance(examples, list) or any(not isinstance(row, dict) or set(row) != {'message', 'category'}
            or any(not isinstance(value, str) for value in row.values()) for row in examples):
        raise ValueError('Invalid runtime examples')
    return {
        'state': {'actualState': copy.deepcopy(state), 'customerRules': question['instructions']},
        'questions': {key: {'type': 'noul',
                           'instructions': {'question': TASK,
                                            'routeKey': key, 'definition': definition},
                           'criteria': dict(CRITERIA)}
                      for key, definition in definitions.items() if key != unknown_key},
    }


def resolve(probabilities, precedence, *, yes, no):
    """Resolve a runtime strict partial order; uncertainty never becomes a guessed route.

    Thresholds are explicit inputs, not defaults or validated operating points.
    An edge [A, B] says A wins when both A and B apply. No semantic extraction,
    probability renormalization, rank-by-score or source-rule compilation occurs.
    """
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError('Membership probabilities required')
    if any(not isinstance(key, str) or not key or type(value) not in {int, float}
           or not math.isfinite(value) or not 0 <= value <= 1 for key, value in probabilities.items()):
        raise ValueError('Invalid membership probabilities')
    if any(type(value) not in {int, float} or not math.isfinite(value) for value in [yes, no]) or not 0 <= no < yes <= 1:
        raise ValueError('Explicit ordered membership cutoffs required')
    if not isinstance(precedence, list):
        raise ValueError('Runtime precedence edges required')
    lower = {key: set() for key in probabilities}
    for edge in precedence:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2 or any(key not in lower for key in edge):
            raise ValueError('Precedence edges must name configured intents')
        lower[edge[0]].add(edge[1])
    for pivot in lower:
        for key in lower:
            if pivot in lower[key]:
                lower[key].update(lower[pivot])
    if any(key in values for key, values in lower.items()):
        raise ValueError('Cyclic precedence is not executable')
    possible = {key for key, value in probabilities.items() if value > no}
    winners = sorted(key for key in possible if not any(key in lower[other] for other in possible))
    if not possible:
        return {'kind': 'clarification', 'reason': 'no-supported-intent'}
    if len(winners) != 1:
        return {'kind': 'clarification', 'reason': 'unresolved-priority'}
    winner = winners[0]
    if probabilities[winner] < yes:
        return {'kind': 'clarification', 'reason': 'uncertain-leading-intent'}
    return {'kind': 'route', 'key': winner}
