from __future__ import annotations

import argparse
import logging
import sys
import time

sys.path.insert(0, "src")

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logging.getLogger("deepgit.pipeline").setLevel(logging.INFO)
logging.getLogger("deepgit.control").setLevel(logging.INFO)

from bench.dataset import CASES, EDGE_CASES, HELDOUT_CASES
from bench.metrics import CaseScore, aggregate, anti_at_top1, anti_in_topk, hit_at_k, reciprocal_rank


def _safe(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def run_core(cases, adaptive: bool) -> list[CaseScore]:
    from deepgit import search_structured

    scores: list[CaseScore] = []
    for case in cases:
        print(f"\n=== [{case.id}] {_safe(case.intent)[:80]} ...")
        try:
            res = search_structured(case.intent, adaptive=adaptive)
        except Exception as exc:  # keep the suite running; a crash is a failure
            print(f"    ERROR: {exc!r}")
            scores.append(
                CaseScore(case.id, False, False, False, 0.0, False, 0,
                          f"ERROR:{exc!r}", False, 0, 0.0)
            )
            continue
        ranked = res.top_repos(10)
        score = CaseScore(
            id=case.id,
            hit1=hit_at_k(ranked, case.relevant, 1),
            hit3=hit_at_k(ranked, case.relevant, 3),
            hit5=hit_at_k(ranked, case.relevant, 5),
            mrr=reciprocal_rank(ranked, case.relevant),
            anti_top1=anti_at_top1(ranked, case.anti),
            anti_in_top3=anti_in_topk(ranked, case.anti, 3),
            top_repo=ranked[0] if ranked else "(none)",
            escalated=res.escalated,
            llm_calls=res.llm_calls,
            elapsed=res.elapsed,
        )
        scores.append(score)
        flag = "HIT" if score.hit3 else "miss"
        anti = " ANTI@1!" if score.anti_top1 else ""
        print(
            f"    -> #1={score.top_repo} [{flag}] mrr={score.mrr:.2f}"
            f" esc={score.escalated} calls={score.llm_calls}{anti}"
        )
    return scores


def run_edge(cases) -> list[dict]:
    from deepgit import search_structured

    results: list[dict] = []
    for case in cases:
        print(f"\n=== EDGE [{case.id}] {_safe(case.intent)[:80]}")
        ok = True
        detail = ""
        try:
            res = search_structured(case.intent)
            top = res.top
            detail = (
                f"scanned={res.candidates_scanned} ranked={len(res.ranked)} "
                f"top={top.repo if top else '(none)'} "
                f"verdict={top.verdict if top else '-'}"
            )
            # Robustness contract: never crash; always return a SearchResult;
            # if it returned repos, the top must carry a verdict string.
            if res.ranked and not top.verdict:
                ok = False
                detail += " [missing verdict]"
        except Exception as exc:
            ok = False
            detail = f"CRASHED: {exc!r}"
        results.append({"id": case.id, "ok": ok, "detail": detail, "exp": case.expectation})
        print(f"    -> {'OK' if ok else 'FAIL'} :: {_safe(detail)}")
    return results


def write_report(scores, edge_results, adaptive: bool) -> None:
    lines = ["# DeepGit-Bench report", ""]
    lines.append(f"_Adaptive controller: {'ON' if adaptive else 'OFF (fast lane only)'}_")
    lines.append("")
    if scores:
        agg = aggregate(scores)
        lines.append("## Aggregate")
        lines.append(agg.as_table())
        lines.append("## Per-case")
        lines.append("| case | #1 pick | Hit@1 | Hit@3 | MRR | anti@1 | escalated | calls | secs |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for s in scores:
            lines.append(
                f"| {s.id} | {s.top_repo} | {'Y' if s.hit1 else '·'} | "
                f"{'Y' if s.hit3 else '·'} | {s.mrr:.2f} | "
                f"{'FAIL' if s.anti_top1 else '·'} | {'Y' if s.escalated else '·'} | "
                f"{s.llm_calls} | {s.elapsed:.0f} |"
            )
        lines.append("")
    if edge_results:
        lines.append("## Edge / robustness suite")
        lines.append("| case | result | detail | expectation |")
        lines.append("|---|---|---|---|")
        for r in edge_results:
            lines.append(
                f"| {r['id']} | {'PASS' if r['ok'] else 'FAIL'} | {r['detail']} | {r['exp']} |"
            )
        lines.append("")
    with open("_bench_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n-> wrote _bench_report.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit to first N core cases")
    ap.add_argument("--case", type=str, default="", help="run a single core case by id")
    ap.add_argument("--edge", action="store_true", help="run the edge/robustness suite")
    ap.add_argument("--heldout", action="store_true", help="run the held-out set (unseen)")
    ap.add_argument("--no-adaptive", action="store_true", help="ablation: gate off")
    args = ap.parse_args()

    adaptive = not args.no_adaptive
    t0 = time.time()

    scores: list[CaseScore] = []
    edge_results: list[dict] = []

    if args.edge:
        edge_results = run_edge(EDGE_CASES)
    else:
        cases = HELDOUT_CASES if args.heldout else CASES
        if args.case:
            cases = [c for c in cases if c.id == args.case]
        elif args.n:
            cases = cases[: args.n]
        scores = run_core(cases, adaptive)

    write_report(scores, edge_results, adaptive)

    if scores:
        agg = aggregate(scores)
        print("\n" + agg.as_table())
    if edge_results:
        passed = sum(r["ok"] for r in edge_results)
        print(f"\nEdge suite: {passed}/{len(edge_results)} passed")

    print(f"\nTotal bench time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
    print("OK")
