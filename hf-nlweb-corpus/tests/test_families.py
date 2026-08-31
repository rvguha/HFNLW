from hfcorpus.families import family_id_for, resolve, variants_by_family


def record(repo_id, *, downloads=0, card_sha="", base=(), architecture="DemoForCausalLM",
           parameters=1_000_000, created="2025-01-01T00:00:00.000Z"):
    return {
        "item": {
            "@id": f"https://huggingface.co/{repo_id}",
            "dateCreated": created,
            "additionalProperty": [
                {"@type": "PropertyValue", "propertyID": "huggingface:downloads",
                 "name": "Downloads", "value": downloads},
                {"@type": "PropertyValue", "propertyID": "ml:architecture",
                 "name": "Architecture", "value": architecture},
                {"@type": "PropertyValue", "propertyID": "ml:parameterCount",
                 "name": "Parameter count", "value": parameters},
            ],
            "description": "A model for things. " * 20,
        },
        "internal": {"repo_id": repo_id, "clean_card_sha256": card_sha,
                     "declared_base_models": list(base)},
    }


def test_declared_lineage_groups_a_family(patterns):
    records = [record("acme/base-1b"), record("acme/base-1b-instruct", base=["acme/base-1b"]),
               record("someone/base-1b-medical", base=["acme/base-1b-instruct"])]
    out = resolve(records, patterns)
    assert len({a.family_id for a in out.values()}) == 1
    assert all(a.discovery_scope == "primary" for a in out.values())


def test_identical_cards_and_config_collapse_to_one_canonical(patterns):
    records = [record("acme/model", downloads=1000, card_sha="abc"),
               record("copycat/model", downloads=5, card_sha="abc")]
    out = resolve(records, patterns)
    assert out["copycat/model"].relation == "exact_duplicate_of"
    assert out["copycat/model"].canonical_repo_id == "acme/model"
    assert out["copycat/model"].discovery_scope == "variant"
    assert out["acme/model"].relation == "canonical"


def test_identical_cards_with_different_configs_are_not_merged(patterns):
    records = [record("acme/model", card_sha="abc", architecture="A", parameters=1),
               record("other/model", card_sha="abc", architecture="B", parameters=2)]
    out = resolve(records, patterns)
    assert out["other/model"].relation != "exact_duplicate_of"


def test_suppressed_packaging_variants_attach_without_becoming_results(patterns):
    records = [record("acme/model-1b", downloads=10)]
    out = resolve(records, patterns, variants=[{"repo_id": "acme/model-1b-GGUF"}])
    variant = out["acme/model-1b-GGUF"]
    assert variant.discovery_scope == "variant"
    assert variant.family_id == out["acme/model-1b"].family_id
    assert variant.canonical_repo_id == "acme/model-1b"
    assert variants_by_family(out)[variant.family_id] == ["acme/model-1b-GGUF"]


def test_a_variant_whose_parent_is_absent_does_not_invent_a_canonical_link(patterns):
    out = resolve([], patterns, variants=[{"repo_id": "someone/unknown-model-GGUF"}])
    variant = out["someone/unknown-model-GGUF"]
    assert variant.canonical_repo_id == ""
    assert any("not represented" in e for e in variant.evidence)


def test_family_id_is_stable_across_packaging_variants(patterns):
    assert family_id_for("Qwen/Qwen3-8B-GGUF", patterns) == family_id_for("Qwen/Qwen3-8B", patterns)
