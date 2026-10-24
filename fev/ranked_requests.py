"""Bound the membership work while keeping unqueried routes explicitly unknown."""
import re

from fev.intent_membership import request_for
from fev.ranked_membership import resolve


def source_precedence(request):
    """Only the exact explicit-pair syntax in our authored synthetic fixtures.

    This is not an arbitrary-DSL or general natural-language policy compiler.
    Other complete source rules remain the original Choice judgment's input.
    """
    question = request['questions']['route']
    matches = re.findall(r'если одновременно выражены цели «([^»]+)» и «([^»]+)», основной является цель «([^»]+)»',
                         question['instructions'])
    meanings = {}
    for key, definition in question['criteria'].items():
        meanings.setdefault(definition, []).append(key)
    edges = []
    for first, second, preferred in matches:
        if first == second or preferred not in {first, second} or any(len(meanings.get(value, [])) != 1
                                                                   for value in [first, second, preferred]):
            raise ValueError('Explicit pair preference has ambiguous route references')
        other = second if preferred == first else first
        edges.append([meanings[preferred][0], meanings[other][0]])
    return edges


def prepare(request, choice, *, unknown_key, maximum_questions=3):
    if type(maximum_questions) is not int or not 1 <= maximum_questions <= 100:
        raise ValueError('A bounded membership question count is required')
    full = request_for(request, unknown_key)
    if set(choice) != set(request['questions']['route']['criteria']):
        raise ValueError('Native Choice must match the original runtime configuration')
    edges = source_precedence(request)
    evidence = dict.fromkeys(full['questions'])
    # Validate all identities/distributions/precedence before using rank or skipping.
    resolve(evidence, choice, edges, unknown_key=unknown_key, yes=.8, no=.2)
    best = max(choice.values())
    if choice[unknown_key] == best:
        return {'request': None, 'queriedKeys': [], 'unqueriedKeys': list(evidence),
                'precedence': edges, 'reason': 'native-unknown-cannot-lose-to-an-intent'}
    ranked = sorted(evidence, key=lambda key: choice[key], reverse=True)[:maximum_questions]
    selected = set(ranked)
    # Retain the configured question order; this changes the native sequence and
    # therefore requires new inference, not assumed equality to old full fan-out.
    full['questions'] = {key: value for key, value in full['questions'].items() if key in selected}
    return {'request': full, 'queriedKeys': list(full['questions']),
            'unqueriedKeys': [key for key in evidence if key not in selected], 'precedence': edges,
            'reason': 'bounded-membership-with-robust-unqueried-evidence'}
