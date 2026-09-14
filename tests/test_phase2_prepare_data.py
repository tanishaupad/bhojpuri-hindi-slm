from __future__ import annotations

from pathlib import Path

import numpy as np
import sentencepiece as spm

from lma_phase1.io import write_jsonl
from lma_phase1.schema import Document
from lma_phase1.tokenizer.common import normalize_tokenizer_input
from lma_phase2.prepare_data import DTYPE, checkpoint_state, prepare_split


def document(document_id: str, text: str) -> Document:
    return Document(
        document_id=document_id,
        language="bho",
        text=text,
        source_name="test",
        collection_type="manual",
        collection_method="test",
        license="test",
        collected_at="2026-08-23",
    )


def test_prepare_split_resumes_at_durable_document_boundary(tmp_path) -> None:
    tokenizer_path = Path("bhojpuri/tokenizer/bhojpuri.model").resolve()
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    documents = [
        document("first", "ई पहिला भोजपुरी परीक्षण वाक्य हवे।"),
        document("second", "ई दोसरा भोजपुरी परीक्षण वाक्य हवे।"),
    ]
    input_path = tmp_path / "input.jsonl.gz"
    write_jsonl(input_path, documents)
    output_path = tmp_path / "train.tokens.bin"
    manifest_path = tmp_path / "train.manifest.json"
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    state_path = output_path.with_suffix(output_path.suffix + ".state.json")
    first_tokens = processor.encode(documents[0].text, out_type=int) + [processor.eos_id()]
    np.asarray(first_tokens, dtype=DTYPE).tofile(partial_path)
    checkpoint_state(
        state_path,
        input_path,
        tokenizer_path,
        documents=1,
        tokens=len(first_tokens),
        utf8_bytes=len(documents[0].text.encode("utf-8")),
        last_document_id="first",
    )

    manifest = prepare_split(
        input_path,
        output_path,
        manifest_path,
        tokenizer_path,
        processor,
        language="bho",
        split="train",
        checkpoint_interval=1,
    )

    second_tokens = processor.encode(
        normalize_tokenizer_input(documents[1].text), out_type=int
    ) + [processor.eos_id()]
    actual = np.fromfile(output_path, dtype=DTYPE).tolist()
    assert actual == first_tokens + second_tokens
    assert manifest["document_count"] == 2
    assert manifest["token_count"] == len(actual)
    assert manifest["utf8_bytes"] == sum(len(item.text.encode("utf-8")) for item in documents)
    assert manifest["tokenizer_input_normalization"] == "normalize_tokenizer_input"
    assert not partial_path.exists()
    assert not state_path.exists()
    assert (
        prepare_split(
            input_path,
            output_path,
            manifest_path,
            tokenizer_path,
            processor,
            language="bho",
            split="train",
        )
        == manifest
    )
