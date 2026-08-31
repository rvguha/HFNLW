from pathlib import Path

import yaml

from hfcorpus.select import (
    Candidate,
    _lineage_keys,
    assign_primary_strata,
    deterministic_features,
    score,
    select,
)


def candidate(repo_id, stratum="text_generation_chat", **kwargs):
    defaults = dict(author=repo_id.split("/")[0], sha="0" * 40, pipeline_tag="text-generation",
                    library_name="transformers", downloads=1000, likes=10,
                    last_modified="2026-06-01T00:00:00Z", parameters=1_000_000_000,
                    parameters_source="safetensors")
    defaults.update(kwargs)
    return Candidate(repo_id=repo_id, stratum=stratum, **defaults)


def decide(policy, candidates):
    features = deterministic_features(policy, candidates)
    scores = score(policy, candidates, features)
    return {d.repo_id: d for d in select(policy, candidates, features, scores)}, scores


def test_inaccessible_and_checkpoint_repositories_are_excluded_with_reasons(policy):
    candidates = [candidate("a/private", private=True), candidate("b/gated", gated="manual"),
                  candidate("c/model-checkpoint-500"), candidate("d/fine")]
    decisions, _ = decide(policy, candidates)
    assert decisions["a/private"].reason_code == "excluded_inaccessible"
    assert decisions["b/gated"].reason_code == "excluded_inaccessible"
    assert decisions["c/model-checkpoint-500"].reason_code == "excluded_checkpoint"
    assert decisions["d/fine"].decision == "included"


def test_packaging_variants_are_suppressed_not_deleted(policy):
    decisions, _ = decide(policy, [candidate("a/model"), candidate("a/model-GGUF")])
    assert decisions["a/model-GGUF"].decision == "suppressed"
    assert decisions["a/model-GGUF"].reason_code == "suppressed_variant"


def test_a_single_publisher_cannot_take_the_whole_corpus(policy):
    candidates = [candidate(f"hog/model-{i}", downloads=10_000 - i) for i in range(40)]
    candidates += [candidate(f"other{i}/model", downloads=100) for i in range(40)]
    decisions, _ = decide(policy, candidates)
    included = [d for d in decisions.values() if d.decision == "included"]
    hog = sum(1 for d in included if d.repo_id.startswith("hog/"))
    assert hog <= int(policy.caps["publisher_max_fraction"] * policy.target_count)


def test_derivatives_of_one_base_model_cannot_flood_a_stratum(policy):
    candidates = [candidate(f"tuner{i}/my-tune", base_models=["meta/Llama-3.1-8B"])
                  for i in range(20)]
    decisions, _ = decide(policy, candidates)
    included = sum(1 for d in decisions.values() if d.decision == "included")
    assert included <= policy.caps["family_max_records"]


def test_documentation_and_diversity_lift_a_well_documented_niche_model(policy):
    plain = candidate("big/popular", downloads=1_000_000, likes=900)
    niche = candidate("small/niche", downloads=500, likes=5, languages=["ta"],
                      datasets=["x/y"], license="mit", parameters=200_000_000,
                      eval_results=True, base_models=["big/popular"])
    features = deterministic_features(policy, [plain, niche])
    assert features["small/niche"]["documentation_quality"] > \
        features["big/popular"]["documentation_quality"]
    assert features["small/niche"]["diversity_bonus"] > features["big/popular"]["diversity_bonus"]


def test_every_candidate_gets_exactly_one_decision(policy):
    candidates = [candidate(f"p{i}/m", downloads=i) for i in range(30)]
    decisions, _ = decide(policy, candidates)
    assert len(decisions) == len(candidates)
    assert {d.decision for d in decisions.values()} <= {"included", "excluded", "suppressed"}


def test_a_shared_candidate_goes_to_the_stratum_that_needs_it_most(policy):
    """Otherwise quotas depend on which stratum happens to be listed first, and
    a narrow stratum listed after broad ones is starved of candidates."""
    broad, narrow = policy.strata[0].name, policy.strata[-1].name
    # One candidate only the narrow stratum found, plus a shared one.
    only_narrow = candidate("a/only", stratum=narrow, found_by=[f"{narrow}:q"])
    shared = candidate("b/shared", stratum=broad,
                       found_by=[f"{broad}:q", f"{narrow}:q"])
    # Fill the broad stratum so the narrow one is further from quota.
    filler = [candidate(f"f{i}/m", stratum=broad, found_by=[f"{broad}:q"])
              for i in range(policy.strata[0].quota)]

    pool = [only_narrow, shared, *filler]
    assign_primary_strata(pool, policy)
    assert only_narrow.stratum == narrow
    assert shared.stratum == narrow, "shared candidate should go where it is scarcer"


def test_stratum_assignment_does_not_depend_on_yaml_order(policy):
    a, b = policy.strata[0].name, policy.strata[-1].name
    forward = [candidate("x/m", stratum=a, found_by=[f"{a}:q", f"{b}:q"])]
    backward = [candidate("x/m", stratum=b, found_by=[f"{b}:q", f"{a}:q"])]
    assign_primary_strata(forward, policy)
    assign_primary_strata(backward, policy)
    assert forward[0].stratum == backward[0].stratum


def test_selection_is_deterministic_for_the_same_pool(policy):
    candidates = [candidate(f"p{i}/m", downloads=1000) for i in range(30)]
    first, _ = decide(policy, candidates)
    second, _ = decide(policy, candidates)
    assert {k: v.decision for k, v in first.items()} == {k: v.decision for k, v in second.items()}


def test_the_family_cap_uses_the_key_that_ships_with_the_record():
    """The cap was computed one hop up the lineage while families.py resolved to
    the root, so 24 records shipped under one family against a cap of 4."""
    patterns = yaml.safe_load(Path("config/selection.yaml").read_text())["patterns"]

    chain = [
        candidate("qwen/Qwen2.5-7B"),
        candidate("qwen/Qwen2.5-7B-Instruct", base_models=["qwen/Qwen2.5-7B"]),
        candidate("qwen/Qwen2.5-Coder-7B", base_models=["qwen/Qwen2.5-7B"]),
        candidate("other/tune", base_models=["qwen/Qwen2.5-7B-Instruct"]),
        candidate("third/deep-tune", base_models=["other/tune"]),
    ]
    keys = _lineage_keys(chain, patterns)
    assert len(set(keys.values())) == 1, f"whole chain should share one key, got {keys}"


def test_derivatives_of_a_deep_chain_cannot_flood_the_corpus(policy):
    chain = [candidate("root/base")]
    for i in range(20):
        parent = "root/base" if i == 0 else f"t{i-1}/tune"
        chain.append(candidate(f"t{i}/tune", base_models=[parent]))
    decisions, _ = decide(policy, chain)
    included = sum(1 for d in decisions.values() if d.decision == "included")
    assert included <= policy.caps["family_max_records"]
