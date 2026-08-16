from lma_phase1.tokenizer.common import normalize_tokenizer_input


def test_normalize_tokenizer_input_collapses_unicode_whitespace() -> None:
    assert normalize_tokenizer_input(" भोजपुरी\n\nभाषा\tबा ") == "भोजपुरी भाषा बा"
