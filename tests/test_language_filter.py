from lma_phase1.language_filter import (
    build_models,
    calibrated_thresholds,
    normalize_for_language_id,
    percentile,
    score_difference,
)


def test_normalize_for_language_id_removes_non_devanagari() -> None:
    assert normalize_for_language_id("हमार भाषा Bhojpuri 123 बा") == "हमार भाषा बा"


def test_ngram_score_separates_repeated_seed_styles() -> None:
    bhojpuri = [
        "हमनी के गाँव में आजु मेला लागल बा आ सभे खुश बाड़े। " * 8,
        "रउरा कहाँ जात बानी आ उहाँ का करत बानी। " * 8,
    ]
    hindi = [
        "हमारे गाँव में आज मेला लगा है और सभी लोग खुश हैं। " * 8,
        "आप कहाँ जा रहे हैं और वहाँ क्या कर रहे हैं। " * 8,
    ]
    bho_model, hi_model = build_models(bhojpuri, hindi, size=4)

    assert score_difference(bhojpuri[0], bho_model, hi_model, 4) > 0
    assert score_difference(hindi[0], bho_model, hi_model, 4) < 0


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.0


def test_calibrated_thresholds_use_both_language_distributions() -> None:
    bhojpuri_scores = [0.2] + [0.7] * 9 + [0.8] * 90
    hindi_scores = [-1.0] * 90 + [-0.4] * 9 + [0.1]

    accept, borderline = calibrated_thresholds(bhojpuri_scores, hindi_scores)

    assert accept == 0.7
    assert borderline == 0.2
