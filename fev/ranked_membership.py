"""Experimental composition of independent membership and competing Choice evidence.

Runtime precedence is a strict partial order. Choice breaks the remaining ties;
membership probabilities are never compared to one another as ranking scores.
Only a winner invariant to every uncertain-intent completion may be routed.
This does not infer category relationships or compile arbitrary business prose.
"""
import math

from fev.intent_membership import resolve as validate_membership


def resolve(membership, choice, precedence, *, unknown_key, yes, no):
    if not isinstance(unknown_key, str) or not unknown_key:
        raise ValueError('An explicit unknown route key is required')
    # None means unmeasured, not a fabricated model probability. It spans both
    # presence outcomes in the robustness check, like a measured uncertain value.
    if not isinstance(membership, dict):
        raise ValueError('Configured membership evidence is required')
    evidence = {key: (no + yes) / 2 if value is None else value for key, value in membership.items()}
    # Reuse the frozen primitive's validation, without changing its experiment.
    validate_membership(evidence, precedence, yes=yes, no=no)
    if not isinstance(choice, dict) or unknown_key in membership or set(choice) != set(membership) | {unknown_key}:
        raise ValueError('Choice must cover exactly the configured intents and explicit unknown')
    if any(type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1
           for value in choice.values()) or not math.isclose(sum(choice.values()), 1, abs_tol=1e-5):
        raise ValueError('Invalid competing Choice distribution')

    lower = {key: set() for key in membership}
    for preferred, other in precedence:
        lower[preferred].add(other)
    for pivot in lower:
        for key in lower:
            if pivot in lower[key]:
                lower[key].update(lower[pivot])

    possible = {key for key, value in evidence.items() if value > no}
    certain = {key for key, value in evidence.items() if value >= yes}
    if not certain:
        return {'kind': 'clarification', 'reason': 'no-confirmed-intent'}

    # An intent dominated by an always-present intent cannot win in any completion.
    potential_winners = {key for key in possible if not any(key in lower[other] for other in certain)}
    ranking = {key: choice[key] for key in potential_winners | {unknown_key}}
    best = max(ranking.values())
    winners = [key for key, value in ranking.items() if math.isclose(value, best, abs_tol=1e-12, rel_tol=0)]
    if len(winners) != 1:
        return {'kind': 'clarification', 'reason': 'competing-choice-tie'}
    winner = winners[0]
    if winner == unknown_key:
        return {'kind': 'clarification', 'reason': 'competing-choice-unknown'}
    if winner not in certain:
        return {'kind': 'clarification', 'reason': 'uncertain-potential-winner'}
    # A possible dominator could remove this winner, regardless of its Choice score.
    if any(winner in lower[other] for other in possible):
        return {'kind': 'clarification', 'reason': 'uncertain-priority-could-change-winner'}
    return {'kind': 'route', 'key': winner}
