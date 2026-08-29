from __future__ import annotations

from dataclasses import dataclass


def _norm(name: str) -> str:
    return name.strip().lower()


def _normset(names: set[str]) -> set[str]:
    return {_norm(n) for n in names}


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> bool:
    """True if any relevant repo appears in the top-k."""
    if not relevant:
        return False
    top = _normset(set(ranked[:k]))
    return bool(top & _normset(relevant))


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant repo (0 if none)."""
    if not relevant:
        return 0.0
    rel = _normset(relevant)
    for i, repo in enumerate(ranked, 1):
        if _norm(repo) in rel:
            return 1.0 / i
    return 0.0


def anti_at_top1(ranked: list[str], anti: set[str]) -> bool:
    """True if an anti-pattern repo wrongly landed at #1 (a failure)."""
    if not anti or not ranked:
        return False
    return _norm(ranked[0]) in _normset(anti)


def anti_in_topk(ranked: list[str], anti: set[str], k: int) -> int:
    """Count of anti-pattern repos that leaked into the top-k."""
    if not anti:
        return 0
    return len(_normset(set(ranked[:k])) & _normset(anti))


@dataclass
class CaseScore:
    id: str
    hit1: bool
    hit3: bool
    hit5: bool
    mrr: float
    anti_top1: bool
    anti_in_top3: int
    top_repo: str
    escalated: bool
    llm_calls: int
    elapsed: float


@dataclass
class Aggregate:
    n: int
    hit1: float
    hit3: float
    hit5: float
    mrr: float
    anti_top1_rate: float
    escalation_rate: float
    avg_llm_calls: float
    avg_elapsed: float

    def as_table(self) -> str:
        return (
            f"| metric | value |\n|---|---|\n"
            f"| cases | {self.n} |\n"
            f"| Hit@1 | {self.hit1:.0%} |\n"
            f"| Hit@3 | {self.hit3:.0%} |\n"
            f"| Hit@5 | {self.hit5:.0%} |\n"
            f"| MRR | {self.mrr:.3f} |\n"
            f"| Anti-pattern @#1 (lower=better) | {self.anti_top1_rate:.0%} |\n"
            f"| Escalation rate | {self.escalation_rate:.0%} |\n"
            f"| Avg LLM calls | {self.avg_llm_calls:.1f} |\n"
            f"| Avg latency | {self.avg_elapsed:.1f}s |\n"
        )


def aggregate(scores: list[CaseScore]) -> Aggregate:
    n = len(scores) or 1
    return Aggregate(
        n=len(scores),
        hit1=sum(s.hit1 for s in scores) / n,
        hit3=sum(s.hit3 for s in scores) / n,
        hit5=sum(s.hit5 for s in scores) / n,
        mrr=sum(s.mrr for s in scores) / n,
        anti_top1_rate=sum(s.anti_top1 for s in scores) / n,
        escalation_rate=sum(s.escalated for s in scores) / n,
        avg_llm_calls=sum(s.llm_calls for s in scores) / n,
        avg_elapsed=sum(s.elapsed for s in scores) / n,
    )
