"""Format-independent extraction of supported claims from reasoning continuations.

No reason marker or prescribed answer layout is required. Uninterpreted text stays
visible and is not equated with an incorrect factual assertion.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict

from lma_phase3.evaluate import normalize_answer
from lma_phase3.problems import Problem, closure, solve
from lma_phase3.proof_checker import Claim, claim_supported, parse_claim, sufficient_proof

PREFIXES = (
    "यह बात तो तय है कि ",
    "यह बात सही है कि ",
    "यह स्पष्ट है कि ",
    "यह सही है कि ",
    "क्योंकि ",
    "चूँकि ",
    "इसलिए ",
    "इसलिये ",
    "अतः ",
    "एहसे ",
    "एही से ",
)


def strip_assertive_prefix(text: str) -> str:
    """Remove known affirmative discourse framing, never arbitrary preceding prose."""

    text = normalize_answer(text)
    for prefix in PREFIXES:
        if text.startswith(prefix):
            return strip_assertive_prefix(text[len(prefix) :])
    return text


def natural_claim(text: str, problem: Problem, language: str) -> Claim | None:
    """Recognize complete natural comparison/value clauses without a format delimiter."""

    text = strip_assertive_prefix(text)
    claim = parse_claim(text, problem, language)
    if claim is not None:
        return claim
    names = "|".join(re.escape(normalize_answer(n)) for n in problem.entities)
    lookup = {normalize_answer(n): i for i, n in enumerate(problem.entities)}
    name = f"({names})"
    hi = language == "hi"
    attr = ("उम्र" if hi else "उमिर") if problem.domain == "age" else "लंबाई"
    possessive, copula = ("की", "है") if hi else ("के", "बा")
    greater = "(?:ज़्यादा|ज्यादा|अधिक)" if hi else "(?:ढेर|अधिक)"
    for kind, word in [("max", greater), ("min", "कम")]:
        match = re.fullmatch(f"{name} {possessive} {attr} सबसे {word} {copula}", text)
        if match:
            return Claim(kind, lookup[match[1]], 0)
    high = (
        ("बड़ा" if problem.domain == "age" else "लंबा")
        if hi
        else ("बड़" if problem.domain == "age" else "लमहर")
    )
    low = "छोटा" if hi else "छोट"
    for kind, word in [("gt", high), ("lt", low)]:
        match = re.fullmatch(f"{name} {name} से {word} {copula}", text)
        if match:
            return Claim(kind, lookup[match[1]], lookup[match[2]])
    for kind, word in [("max", high), ("min", low)]:
        match = re.fullmatch(f"{name} सबसे {word} {copula}", text)
        if match:
            return Claim(kind, lookup[match[1]], 0)
    return None


def supported(claim: Claim, problem: Problem) -> bool:
    """Validate parsed claims using supplied values or all consistent rankings."""

    if claim.kind not in ("max", "min"):
        return claim_supported(claim, problem)
    worlds = (
        [problem.values]
        if problem.values
        else [
            w
            for w in itertools.permutations(range(len(problem.entities)))
            if all(w[a] > w[b] for a, b in problem.edges)
        ]
    )
    compare = (lambda a, b: a > b) if claim.kind == "max" else (lambda a, b: a < b)
    return bool(worlds) and all(
        all(i == claim.left or compare(w[claim.left], w[i]) for i in problem.query) for w in worlds
    )


def score_semantics(row: dict, generated: str, base: dict) -> dict:
    """Extract statements and conclusions, preserving unresolved language explicitly."""

    problem = Problem(**row["problem"])
    language = base["language"]
    text = normalize_answer(generated)
    # Labels merely delimit clauses if present; they are not required.
    text = re.sub(r"(?:उत्तर|जवाब|कारण)\s*:", "।", text)
    # Retain question punctuation: asking whether X is true does not assert X.
    clauses = re.findall(r"[^।.!?;\n]+[।.!?;\n]?", text)
    choices = (
        [base["yes"], base["no"]]
        if problem.family == "equality"
        else [problem.entities[i] for i in problem.query]
    )
    choices = {normalize_answer(c) for c in choices}
    explicit = set()
    claims = []
    unknown = []
    recognized = []

    def consume(clause, split_allowed=True):
        clause = clause.strip()
        if clause.endswith("?"):
            unknown.append(clause)
            return
        completed_sentence = clause.endswith(("।", ".", "!", ";"))
        clause = clause.rstrip("।.!;\n").strip()
        if not clause:
            return
        clause = strip_assertive_prefix(clause)
        candidate = clause.removesuffix(" है" if language == "hi" else " बा").strip()
        labelled = bool(re.match(r"^(?:सही उत्तर|उत्तर|जवाब)\s+", candidate))
        candidate = re.sub(r"^(?:सही उत्तर|उत्तर|जवाब)\s+", "", candidate)
        # A name split from "X and Y's age ..." is a subject, not an answer.
        bare_answer = split_allowed and (completed_sentence or len(clauses) == 1)
        if candidate in choices and (bare_answer or labelled):
            explicit.add(candidate)
            recognized.append(clause)
            return
        claim = natural_claim(clause, problem, language)
        if claim is not None:
            claims.append(claim)
            recognized.append(clause)
            return
        pieces = re.split(r"\s+(?:और|आ)\s+|\s*,\s*", clause) if split_allowed else [clause]
        if len(pieces) > 1:
            for piece in pieces:
                consume(piece, False)
        else:
            unknown.append(clause)

    for clause in clauses:
        consume(clause)
    supported_flags = [supported(c, problem) for c in claims]
    false_claims = [asdict(c) for c, ok in zip(claims, supported_flags, strict=True) if not ok]
    entailed = set()
    edges = {
        (c.left, c.right) if c.kind == "gt" else (c.right, c.left)
        for c in claims
        if c.kind in ("gt", "lt")
    }
    contradictory = False
    try:
        reach = closure(len(problem.entities), tuple(edges))
    except ValueError:
        reach = None
        contradictory = True
    if problem.family == "equality":
        for c in claims:
            if {c.left, c.right} == {0, 1} and c.kind in ("eq", "ne"):
                entailed.add(normalize_answer(base["yes" if c.kind == "eq" else "no"]))
    else:
        direction = problem.direction
        for c in claims:
            if c.kind == ("max" if direction == "greater" else "min") and c.left in problem.query:
                entailed.add(normalize_answer(problem.entities[c.left]))
        if reach is not None:
            for i in problem.query:
                if all(
                    i == j or (reach[i][j] if direction == "greater" else reach[j][i])
                    for j in problem.query
                ):
                    entailed.add(normalize_answer(problem.entities[i]))
    answers = explicit | entailed
    # A trailing bare name is interpretable when the preceding comparisons already
    # establish it; a repeated subject after unrelated prose remains unresolved.
    confirmed_fragments = [s for s in unknown if normalize_answer(s) in entailed]
    recognized.extend(confirmed_fragments)
    unknown = [s for s in unknown if s not in confirmed_fragments]
    reference = normalize_answer(row["answer_text"])
    answer_status = (
        "conflicting"
        if len(answers) > 1 or contradictory
        else "unresolved"
        if not answers
        else "correct"
        if reference in answers
        else "incorrect"
    )
    ordinary = [c for c in claims if c.kind not in ("max", "min")]
    sufficient = False
    if claims and not false_claims and not contradictory:
        sufficient = sufficient_proof(ordinary, problem)
        if not sufficient and problem.values and problem.family in ("direct", "extremum"):
            gold = solve(problem)
            sufficient = {c.left for c in ordinary if c.kind == "value"} == set(
                range(len(problem.entities))
            ) and any(
                c.kind == ("max" if problem.direction == "greater" else "min") and c.left == gold
                for c in claims
            )
    full = bool(recognized) and not unknown
    return {
        "answer_status": answer_status,
        "answer_correct": answer_status == "correct",
        "answer_resolved": answer_status in ("correct", "incorrect"),
        "explicit_answers": sorted(explicit),
        "conclusions_from_comparisons": sorted(entailed),
        "claim_coverage": bool(claims),
        "supported_claim_found": any(supported_flags),
        "recognized_claims": [asdict(c) for c in claims],
        "unsupported_claims": false_claims,
        "has_unsupported_claim": bool(false_claims),
        "uninterpreted_clauses": unknown,
        "full_text_coverage": full,
        "sufficient_supported_justification": sufficient,
        "explanation_fully_verified": sufficient and full,
        "joint_supported": answer_status == "correct" and sufficient,
        "joint_fully_verified": answer_status == "correct" and sufficient and full,
        "note": "Unknown clauses remain unverified; supported-subset results do not validate them.",
    }
