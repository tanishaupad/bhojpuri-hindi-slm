"""Symbolic comparison problems; labels depend only on exposed facts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Problem:
    """Entity-indexed numerical facts or strict greater-than edges."""

    family: str
    domain: str
    entities: tuple[str, ...]
    values: tuple[int, ...] = ()
    edges: tuple[tuple[int, int], ...] = ()
    query: tuple[int, ...] = ()
    direction: str = "greater"


def closure(size: int, edges: tuple[tuple[int, int], ...]) -> list[list[bool]]:
    """Compute strict transitive reachability, rejecting cycles and invalid endpoints."""

    reachable = [[False] * size for _ in range(size)]
    for left, right in edges:
        if not (0 <= left < size and 0 <= right < size):
            raise ValueError("Invalid entity index")
        reachable[left][right] = True
    for middle in range(size):
        for left in range(size):
            for right in range(size):
                reachable[left][right] |= reachable[left][middle] and reachable[middle][right]
    if any(reachable[i][i] for i in range(size)):
        raise ValueError("Contradictory strict comparisons")
    return reachable


def solve(problem: Problem) -> int | bool:
    """Return the answer entity index, or equality label; reject ambiguous problems."""

    size = len(problem.entities)
    if size not in (2, 3) or len(set(problem.entities)) != size:
        raise ValueError("Expected two or three distinct entities")
    if problem.direction not in ("greater", "smaller"):
        raise ValueError("Invalid direction")
    if problem.family == "transitivity":
        if len(problem.query) != 2 or len(set(problem.query)) != 2:
            raise ValueError("Expected two distinct queried endpoints")
        left, right = problem.query
        if not all(0 <= i < size for i in problem.query):
            raise ValueError("Invalid query endpoint")
        reachable = closure(size, problem.edges)
        if not reachable[left][right] and not reachable[right][left]:
            raise ValueError("Premises do not determine the answer")
        greater, smaller = (left, right) if reachable[left][right] else (right, left)
        return greater if problem.direction == "greater" else smaller
    if len(problem.values) != size or problem.edges:
        raise ValueError("Numeric problems require one exposed value per entity")
    if problem.family == "equality":
        if size != 2:
            raise ValueError("Equality requires two entities")
        return problem.values[0] == problem.values[1]
    if problem.family not in ("direct", "extremum"):
        raise ValueError("Unknown task family")
    if size != (2 if problem.family == "direct" else 3):
        raise ValueError("Wrong entity count for task")
    best = (max if problem.direction == "greater" else min)(problem.values)
    if problem.values.count(best) != 1:
        raise ValueError("Non-unique extremum")
    return problem.values.index(best)


def verify_two_hop(problem: Problem) -> None:
    """Require two necessary edges and no direct answer-revealing endpoint edge."""

    if problem.family != "transitivity" or len(problem.edges) != 2:
        raise ValueError("Expected a two-premise transitivity problem")
    solve(problem)
    if any(set(edge) == set(problem.query) for edge in problem.edges):
        raise ValueError("Query relation is directly exposed")
    for edge in problem.edges:
        reduced = Problem(**{**asdict(problem), "edges": (edge,)})
        try:
            solve(reduced)
        except ValueError:
            continue
        raise ValueError("Both premises must be necessary")


def world_fingerprint(problem: Problem) -> str:
    """Group all questions and premise permutations about the same named world.

    Language, family, query and direction are deliberately excluded. Reordered
    entity arrays are canonicalized by name. Future releases must exclude pilot worlds.
    """

    if problem.edges:
        reachable = closure(len(problem.entities), problem.edges)
        facts = sorted(
            (problem.entities[i], problem.entities[j])
            for i in range(len(problem.entities))
            for j in range(len(problem.entities))
            if reachable[i][j]
        )
    else:
        facts = sorted(zip(problem.entities, problem.values, strict=True))
    payload = json.dumps([problem.domain, facts], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
