from hfcorpus.naming import (
    humanize,
    inferred_parameters_from_name,
    provisional_family_key,
    strip_variant_tokens,
    variant_kind,
)


def test_quantization_and_format_suffixes_are_recognised(patterns):
    assert variant_kind("TheBloke/Mistral-7B-GGUF", patterns) == "quantization"
    assert variant_kind("x/Model-Q4_K_M-GGUF", patterns) == "quantization"
    assert variant_kind("x/Model-IQ3_XXS", patterns) == "quantization"
    assert variant_kind("onnx-community/whisper-tiny-onnx", patterns) == "format_conversion"
    assert variant_kind("x/model-checkpoint-500", patterns) == "checkpoint"
    assert variant_kind("Qwen/Qwen3-8B", patterns) is None


def test_a_conversion_owner_does_not_make_every_repo_a_conversion(patterns):
    assert variant_kind("onnx-community/Qwen3-8B", patterns) is None


def test_stripping_preserves_separators_so_the_parent_name_stays_real(patterns):
    assert strip_variant_tokens("a/LFM2.5-2.6B-Heretic-GGUF", patterns) == "LFM2.5-2.6B-Heretic"
    assert strip_variant_tokens("a/Mizan-27B-Legal-Q4_K_M-GGUF", patterns) == "Mizan-27B-Legal"


def test_family_key_prefers_declared_lineage(patterns):
    assert provisional_family_key("me/my-llama-tune", ["meta-llama/Llama-3.1-8B"], patterns) == \
        "llama-3.1-8b"


def test_parameter_counts_parsed_from_names_are_only_a_fallback():
    assert inferred_parameters_from_name("meta-llama/Llama-3.1-8B-Instruct") == 8_000_000_000
    assert inferred_parameters_from_name("google/embeddinggemma-300m") == 300_000_000
    assert inferred_parameters_from_name("openai-community/gpt2") is None


def test_humanize_keeps_decimal_sizes_intact():
    assert humanize("Qwen/Qwen3-0.6B") == "Qwen3 0.6B"
