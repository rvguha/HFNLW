"""A reference set is only worth what its failure handling is worth.

The run that motivated these tests lost 38 of 45 queries: OpenRouter started
returning 402, the server degraded to returning no results, and every one was
recorded as "no relevant records found" -- indistinguishable, in the output file,
from a judge that read 200 records and rejected them all.
"""

from hfcorpus.goldenset import Judgement


def stub_endpoint(script):
    """Replace judge_query with a scripted sequence of Judgements."""
    import hfcorpus.goldenset as gs
    calls = iter(script)

    def fake(endpoint, question, judge, width, timeout=0):
        return next(calls)

    gs.judge_query = fake
    return gs


def relevant(judge, urls, cost=2.5):
    return Judgement("", judge, [{"url": u, "name": u, "score": 90} for u in urls],
                     retrieved=200, cost_usd=cost, tokens=300_000)


QUERIES = [{"id": "q1", "question": "one"}]
JUDGES = ["a/model", "b/model"]


def test_agreement_is_the_intersection_of_what_both_judges_accepted():
    gs = stub_endpoint([relevant("a/model", ["u1", "u2", "u3"]),
                        relevant("b/model", ["u2", "u3", "u4"])])
    out = gs.build("http://x/ask", QUERIES, JUDGES, 200, log=lambda *_: None)
    q = out["per_query"][0]
    assert q["agreed"] == ["u2", "u3"]
    assert q["union"] == ["u1", "u2", "u3", "u4"]
    assert q["agreement"] == 0.5
    assert out["incomplete"] == []


def test_a_judge_that_was_never_billed_is_an_error_not_a_verdict():
    """The 402 case: no calls, no cost, an empty list that looks like a decision."""
    unbilled = Judgement("", "b/model", [], retrieved=200, cost_usd=0.0, tokens=0)
    gs = stub_endpoint([relevant("a/model", ["u1", "u2"]), unbilled])
    out = gs.build("http://x/ask", QUERIES, JUDGES, 200, log=lambda *_: None)
    assert out["incomplete"] == ["q1"], "a failed judge must mark the query incomplete"


def test_a_failed_judge_does_not_shrink_the_agreed_set():
    """Treating an empty failure as a vote would make agreement look unanimous
    that nothing is relevant, and silently empty the reference set."""
    unbilled = Judgement("", "b/model", [], retrieved=200, cost_usd=0.0, tokens=0)
    gs = stub_endpoint([relevant("a/model", ["u1", "u2"]), unbilled])
    out = gs.build("http://x/ask", QUERIES, JUDGES, 200, log=lambda *_: None)
    q = out["per_query"][0]
    assert set(q["agreed"]) == {"u1", "u2"}, "the surviving judge's verdict stands alone"


def test_an_incomplete_query_is_excluded_from_mean_agreement():
    gs = stub_endpoint([relevant("a/model", ["u1"]),
                        Judgement("", "b/model", [], cost_usd=0.0, tokens=0)])
    out = gs.build("http://x/ask", QUERIES, JUDGES, 200, log=lambda *_: None)
    assert out["mean_agreement"] is None


def test_a_genuine_empty_verdict_is_reported_separately_from_a_failure():
    """Both judges ran, were billed, and found nothing. That is a real result."""
    gs = stub_endpoint([relevant("a/model", [], cost=2.1), relevant("b/model", [], cost=2.6)])
    out = gs.build("http://x/ask", QUERIES, JUDGES, 200, log=lambda *_: None)
    assert out["incomplete"] == []
    assert out["queries_with_no_relevant_record"] == ["q1"]


def test_a_bought_judgement_is_not_bought_twice(tmp_path):
    """Overwriting a reference set cost this project $31.73 of paid verdicts."""
    from hfcorpus.goldenset import judgement_key
    from hfcorpus.store import Store

    store = Store(tmp_path)
    gs = stub_endpoint([relevant("a/model", ["u1", "u2"])])
    first = gs.build("http://x/ask", QUERIES, ["a/model"], 200, store=store, corpus="snap",
                     log=lambda *_: None)
    assert first["total_cost_usd"] > 0
    assert store.judgement(judgement_key("one", "a/model", 200, "snap")).exists()

    # Second run: the stub would raise StopIteration if it were called again.
    second = gs.build("http://x/ask", QUERIES, ["a/model"], 200, store=store, corpus="snap",
                      log=lambda *_: None)
    assert second["total_cost_usd"] == 0
    assert second["judgements_reused_from_cache"] == 1
    assert second["per_query"][0]["union"] == ["u1", "u2"]


def test_a_failed_judgement_is_never_cached(tmp_path):
    """Caching a 402 would make the failure permanent and free."""
    from hfcorpus.store import Store

    store = Store(tmp_path)
    unbilled = Judgement("", "a/model", [], retrieved=200, cost_usd=0.0, tokens=0)
    gs = stub_endpoint([unbilled, relevant("a/model", ["u1"])])
    gs.build("http://x/ask", QUERIES, ["a/model"], 200, store=store, corpus="snap",
             log=lambda *_: None)
    retried = gs.build("http://x/ask", QUERIES, ["a/model"], 200, store=store, corpus="snap",
                       log=lambda *_: None)
    assert retried["per_query"][0]["union"] == ["u1"], "a failure must be retried, not remembered"


def test_the_key_separates_purchases_that_differ(tmp_path):
    from hfcorpus.goldenset import judgement_key
    base = judgement_key("q", "a/model", 200, "snap")
    assert base != judgement_key("q2", "a/model", 200, "snap")
    assert base != judgement_key("q", "b/model", 200, "snap")
    assert base != judgement_key("q", "a/model", 100, "snap")
    assert base != judgement_key("q", "a/model", 200, "other-snap")
