from dataclasses import replace

import pytest

from lma_phase3.problems import Problem, solve, verify_two_hop, world_fingerprint


def test_ambiguous_fork_is_not_a_chain():
    problem = Problem("transitivity", "age", ("A", "B", "C"), edges=((0, 1), (0, 2)), query=(1, 2))
    with pytest.raises(ValueError, match="do not determine"):
        solve(problem)


def test_contradiction_and_direct_shortcut_rejected():
    problem = Problem("transitivity", "age", ("A", "B", "C"), edges=((0, 1), (1, 0)), query=(0, 1))
    with pytest.raises(ValueError, match="Contradictory"):
        solve(problem)
    with pytest.raises(ValueError, match="directly exposed"):
        verify_two_hop(replace(problem, edges=((0, 1), (1, 2))))


def test_both_chain_premises_necessary_and_order_invariant():
    problem = Problem(
        "transitivity",
        "height",
        ("A", "B", "C"),
        edges=((2, 0), (0, 1)),
        query=(1, 2),
        direction="smaller",
    )
    verify_two_hop(problem)
    assert solve(problem) == 1
    assert solve(replace(problem, edges=problem.edges[::-1])) == 1
    assert solve(replace(problem, direction="greater")) == 2


def test_ties_equality_and_relabeling():
    problem = Problem("direct", "age", ("A", "B"), values=(10, 10), query=(0, 1))
    with pytest.raises(ValueError, match="Non-unique"):
        solve(problem)
    assert solve(replace(problem, family="equality")) is True
    assert solve(replace(problem, family="equality", values=(10, 11))) is False
    assert solve(replace(problem, entities=("X", "Y"), values=(10, 11))) == 1


def test_world_group_survives_question_entity_and_premise_reordering():
    a = Problem("transitivity", "age", ("A", "B", "C"), edges=((0, 1), (1, 2)), query=(0, 2))
    b = Problem(
        "transitivity",
        "age",
        ("C", "A", "B"),
        edges=((2, 0), (1, 2)),
        query=(0, 1),
        direction="smaller",
    )
    assert world_fingerprint(a) == world_fingerprint(b)
    numeric = Problem("direct", "age", ("A", "B"), values=(10, 12))
    assert world_fingerprint(numeric) == world_fingerprint(
        replace(numeric, entities=("B", "A"), values=(12, 10), direction="smaller")
    )
