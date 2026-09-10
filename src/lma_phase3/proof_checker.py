"""Deterministic verification of controlled Hindi/Bhojpuri reasoning explanations.

Recognized claims are checked against supplied facts and must contain sufficient
justification. Unsupported language is unverified, not automatically a false claim.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from lma_phase3.evaluate import normalize_answer
from lma_phase3.problems import Problem, closure, solve


@dataclass(frozen=True)
class Claim:
    """An entity value or a binary relation parsed from a complete sentence."""

    kind: str
    left: int
    right: int  # A numeric value for value claims; otherwise an entity index.


def parse_claim(sentence: str, problem: Problem, language: str) -> Claim | None:
    """Recognize anchored factual sentences; never match an affirmative substring in negation."""

    sentence = normalize_answer(sentence)
    sentence = re.sub(r"^(?:इसलिए|इसलिये|अतः|एहसे|एही से)\s*,?\s*", "", sentence)
    names = "|".join(re.escape(normalize_answer(n)) for n in problem.entities)
    lookup = {normalize_answer(n): i for i, n in enumerate(problem.entities)}
    name = f"({names})"
    hindi = language == "hi"
    attribute = ("उम्र" if hindi else "उमिर") if problem.domain == "age" else "लंबाई"
    possessive, copula = ("की", "है") if hindi else ("के", "बा")
    unit = ("साल" if hindi else "बरिस") if problem.domain == "age" else "सेंटीमीटर"
    match = re.fullmatch(f"{name} {possessive} {attribute} ([0-9०-९]+) {unit} {copula}", sentence)
    if match:
        return Claim("value", lookup[match[1]], int(match[2]))
    greater = "(?:ज़्यादा|ज्यादा|अधिक)" if hindi else "(?:ढेर|अधिक)"
    for relation, wording in [("gt", greater), ("lt", "कम")]:
        match = re.fullmatch(
            f"{name} {possessive} {attribute} {name} से {wording} {copula}", sentence
        )
        if match:
            return Claim(relation, lookup[match[1]], lookup[match[2]])
    for kind, ending in [
        ("eq", f"बराबर {copula}"),
        ("ne", "बराबर नहीं है" if hindi else "बराबर नइखे"),
    ]:
        match = re.fullmatch(
            f"{name} {possessive} {attribute} {name} {possessive} {attribute} के {ending}", sentence
        )
        if match:
            return Claim(kind, lookup[match[1]], lookup[match[2]])
        plural = "दोनों" if hindi else "दूनो"
        if len(problem.entities) == 2 and sentence == f"{plural} {possessive} {attribute} {ending}":
            return Claim(kind, 0, 1)
    return None


def claim_supported(claim: Claim, problem: Problem) -> bool:
    """Check facts literally and relations in every world consistent with the premises."""

    if claim.kind == "value":
        return bool(problem.values) and problem.values[claim.left] == claim.right
    worlds = (
        [problem.values]
        if problem.values
        else [
            world
            for world in itertools.permutations(range(len(problem.entities)))
            if all(world[a] > world[b] for a, b in problem.edges)
        ]
    )
    if not worlds:
        return False

    def holds(world):
        a, b = world[claim.left], world[claim.right]
        return {"gt": a > b, "lt": a < b, "eq": a == b, "ne": a != b}[claim.kind]

    return all(holds(world) for world in worlds)


def sufficient_proof(claims: list[Claim], problem: Problem) -> bool:
    """Require grounded values/comparisons or both links and the transitive conclusion."""

    edges = {
        (c.left, c.right) if c.kind == "gt" else (c.right, c.left)
        for c in claims
        if c.kind in ("gt", "lt")
    }
    answer = solve(problem)
    if problem.family == "transitivity":
        if not set(map(tuple, problem.edges)) <= edges:
            return False
        a, b = problem.query
        greater = answer if problem.direction == "greater" else (b if answer == a else a)
        smaller = b if greater == a else a
        return (greater, smaller) in edges
    if {c.left for c in claims if c.kind == "value"} != set(range(len(problem.entities))):
        return False
    if problem.family == "equality":
        expected = "eq" if answer else "ne"
        return any(c.kind == expected and {c.left, c.right} == {0, 1} for c in claims)
    reachable = closure(len(problem.entities), tuple(edges))
    return all(
        i == answer
        or (reachable[answer][i] if problem.direction == "greater" else reachable[i][answer])
        for i in range(len(problem.entities))
    )


def check_output(row: dict, generated: str, base: dict) -> dict:
    """Report answer accuracy, parsing coverage, proof validity and conjunction separately."""

    problem = Problem(**row["problem"])
    text = normalize_answer(generated)
    cue = base["answer_cue"]
    if text.startswith(cue):
        text = text[len(cue) :].strip()
    reason_count = text.count("कारण:")
    answer_text, explanation = text.split("कारण:", 1) if reason_count == 1 else (text, "")
    answer_text = answer_text.strip().rstrip("।.").strip()
    choices = (
        [base["yes"], base["no"]]
        if problem.family == "equality"
        else [problem.entities[i] for i in problem.query]
    )
    parsed_answer = answer_text if answer_text in [normalize_answer(c) for c in choices] else None
    answer_correct = parsed_answer == normalize_answer(row["answer_text"])
    sentences = [
        normalize_answer(s) for s in re.split(r"[।.!?;\n]+", explanation) if normalize_answer(s)
    ]
    parsed = [parse_claim(sentence, problem, base["language"]) for sentence in sentences]
    unknown = [s for s, c in zip(sentences, parsed, strict=True) if c is None]
    claims = [c for c in parsed if c is not None]
    unsupported = [
        {"sentence": s, "claim": {"kind": c.kind, "left": c.left, "right": c.right}}
        for s, c in zip(sentences, parsed, strict=True)
        if c is not None and not claim_supported(c, problem)
    ]
    coverage = bool(sentences) and not unknown and reason_count == 1
    sufficient = coverage and not unsupported and sufficient_proof(claims, problem)
    status = (
        "missing_explanation"
        if not sentences
        else "unparsed_text"
        if not coverage
        else "unsupported_claim"
        if unsupported
        else "insufficient_justification"
        if not sufficient
        else "verified"
    )
    return {
        "answer_parsed": parsed_answer is not None,
        "answer_correct": answer_correct,
        "explanation_present": bool(sentences),
        "explanation_parseable": coverage,
        "explanation_verified": sufficient,
        "joint_correct": answer_correct and sufficient,
        "status": status,
        "parsed_answer": parsed_answer,
        "unparsed_sentences": unknown,
        "unsupported_claims": unsupported,
        "recognized_claims": len(claims),
    }


def reference_explanation(row: dict, base: dict) -> str:
    """Construct a short sufficient proof; no learned model supplies labels or facts."""

    problem = Problem(**row["problem"])
    domain = base["domains"][problem.domain]
    sentences = list(row["premises"])
    answer = solve(problem)
    if problem.family == "equality":
        hindi = base["language"] == "hi"
        attribute = ("उम्र" if hindi else "उमिर") if problem.domain == "age" else "लंबाई"
        beginning = f"{'दोनों की' if hindi else 'दूनो के'} {attribute} बराबर"
        ending = ("है" if answer else "नहीं है") if hindi else ("बा" if answer else "नइखे")
        sentences.append(beginning + " " + ending + "।")
    else:
        others = [i for i in problem.query if i != answer]
        for other in others:
            a, b = (answer, other) if problem.direction == "greater" else (other, answer)
            sentences.append(
                domain["relation"].format(left=problem.entities[a], right=problem.entities[b])
            )
    return row["answer_text"] + "। कारण: " + " ".join(sentences)
