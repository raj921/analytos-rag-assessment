"""Runs eval/dataset.json against the live API and reports hit rate, refusal accuracy, leaks, latency.

Usage: .venv/bin/python eval/run_eval.py --name improved.json
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent


def ask(api: str, case: dict) -> dict:
    body = {
        "question": case["question"],
        "history": case.get("history", []),
        "role": case.get("role", "all"),
    }
    req = urllib.request.Request(
        f"{api}/api/ask",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.load(resp)
    payload["_latency_ms"] = int((time.perf_counter() - started) * 1000)
    return payload


REFUSAL = "i don't have enough information"


def evaluate(dataset: list[dict], results: list[dict]) -> dict:
    checked = refused_ok = leaks = keyword_ok = 0
    hits = kw_total = 0
    latencies = []
    tokens = 0
    for case, res in zip(dataset, results):
        retrieved = {c["title"] for c in res.get("retrieved", [])}
        answer = res.get("answer", "").lower()

        if case.get("expected_titles"):
            checked += 1
            if all(t in retrieved for t in case["expected_titles"]):
                hits += 1

        should_refuse = case.get("expect_refusal", False)
        did_refuse = REFUSAL in answer
        if did_refuse == should_refuse:
            refused_ok += 1

        forbidden = set(case.get("must_not_contain_in_retrieval", []))
        if forbidden & retrieved:
            leaks += 1

        for kw in case.get("must_contain", []):
            kw_total += 1
            if kw.lower() in answer:
                keyword_ok += 1

        latencies.append(res["_latency_ms"])
        usage = res.get("usage") or {}
        tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    n = len(dataset)
    return {
        "cases": n,
        "retrieval_hit_rate": round(hits / checked, 3) if checked else None,
        "refusal_accuracy": round(refused_ok / n, 3),
        "access_leaks": leaks,
        "keyword_coverage": round(keyword_ok / kw_total, 3) if kw_total else None,
        "avg_latency_ms": int(sum(latencies) / n),
        "p95_latency_ms": sorted(latencies)[int(0.95 * (n - 1))],
        "total_tokens": tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--name", default=None, help="result file name, e.g. improved.json")
    args = parser.parse_args()

    dataset = json.loads((HERE / "dataset.json").read_text())
    results = []
    failures = []
    for case in dataset:
        try:
            res = ask(args.api, case)
        except Exception as exc:
            print(f"{case['id']}: request failed: {exc}", file=sys.stderr)
            failures.append(case["id"])
            continue
        results.append(res)
        expected = case.get("expected_titles", [])
        got = sorted({c["title"] for c in res.get("retrieved", [])})
        hit = all(t in got for t in expected) if expected else "-"
        print(f"{case['id']:>4} [{case['category']:<14}] hit={hit} "
              f"latency={res['_latency_ms']}ms docs={got}")

    if failures:
        print(f"ERROR: {len(failures)} cases failed: {failures}", file=sys.stderr)
        return 1

    metrics = evaluate(dataset, results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"run-{time.strftime('%Y%m%d-%H%M%S')}.json"
    (out_dir / name).write_text(json.dumps({"metrics": metrics, "cases": results}, indent=2))
    print("\nmetrics:", json.dumps(metrics, indent=2))
    print(f"saved: {out_dir / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
