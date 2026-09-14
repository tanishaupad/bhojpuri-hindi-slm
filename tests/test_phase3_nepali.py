import json
from collections import Counter
from pathlib import Path

import sentencepiece as spm

from lma_phase3.evaluate_proofs import explanation_prompt
from lma_phase3.generate import generate_pilot
from lma_phase3.proof_checker import check_output, reference_explanation

ROOT = Path(__file__).resolve().parents[1]


def test_nepali_generation_and_reference_proofs():
    base = json.loads((ROOT / "nepali/configs/phase3_templates.json").read_text())
    tokenizer = spm.SentencePieceProcessor(
        model_file=str(ROOT / "nepali/tokenizer/nepali.model")
    )
    rows, rejected = generate_pilot(base, tokenizer)

    assert len(rows) == 50
    assert Counter(row["task_family"] for row in rows) == {
        "direct": 13,
        "extremum": 13,
        "equality": 12,
        "transitivity": 12,
    }
    assert all(row["language"] == "ne" for row in rows)
    assert all(row["serialized_tokens_including_eos"] <= 256 for row in rows)
    assert rejected.get("unknown_token", 0) == 0

    for row in rows:
        reference = reference_explanation(row, base)
        assert check_output(row, reference, base)["joint_correct"]
        prompt = explanation_prompt(row, base)
        assert "पहिले उत्तर लेख्नुहोस्" in prompt
        assert tokenizer.unk_id() not in tokenizer.encode(prompt + " " + reference)


def test_nepali_release_has_disjoint_names_and_wording():
    release = json.loads((ROOT / "nepali/configs/phase3_release.json").read_text())
    assert not set(release["core_names"]) & set(release["challenge_names"])
    assert not set(release["core_variants"]) & set(release["challenge_variants"])
    assert sum(release["split_sizes"].values()) == 25_000
