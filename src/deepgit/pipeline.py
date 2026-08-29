from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from deepgit.control import (
    apply_escalation,
    assess_confidence,
    verify_contested,
)
from deepgit.github.graphql import GitHubGraphQL
from deepgit.github.schema import RepoRecord
from deepgit.lm import configure_dspy
from deepgit.reasoning import IntentExpander, RankingReflector, RepoRanker
from deepgit.reasoning.signatures import RankedRepo
from deepgit.retrieval import get_index, intent_terms, prescore, signal_summary

logger = logging.getLogger("deepgit.pipeline")


@dataclass
class SearchResult:
    """Structured output of a DeepGit search — reusable by CLI, MCP, benchmarks."""

    intent: str
    ranked: list[RankedRepo] = field(default_factory=list)
    summary: str = ""
    trace: list[str] = field(default_factory=list)
    candidates_scanned: int = 0
    escalated: bool = False
    llm_calls: int = 0
    elapsed: float = 0.0

    @property
    def top(self) -> RankedRepo | None:
        return self.ranked[0] if self.ranked else None

    def top_repos(self, k: int = 5) -> list[str]:
        return [r.repo for r in self.ranked[:k]]


async def _gather(queries: list[str], per_query: int) -> list[RepoRecord]:
    """Run all search queries concurrently and de-duplicate by full name."""
    async with GitHubGraphQL() as gh:
        batches = await asyncio.gather(
            *(gh.search(q, limit=per_query) for q in queries),
            return_exceptions=True,
        )
    seen: dict[str, RepoRecord] = {}
    for batch in batches:
        if isinstance(batch, Exception):
            logger.warning("search failed: %s", batch)
            continue
        for rec in batch:
            seen.setdefault(rec.name_with_owner, rec)
    return list(seen.values())


def _fallback_queries(terms: list[str], ecosystems: list[str]) -> list[str]:
    """Build short, broad keyword queries from intent terms (GitHub-friendly).

    Keeps GitHub's AND-search from being too narrow while staying on-topic.
    A language qualifier is added when the ecosystem is known.
    """
    terms = [t for t in terms if len(t) > 2][:6]
    if not terms:
        return []
    lang = f" language:{ecosystems[0].lower()}" if ecosystems else ""
    queries: list[str] = [f"{t}{lang}" for t in terms[:3]]
    for i in range(min(2, len(terms) - 1)):
        queries.append(f"{terms[i]} {terms[i + 1]}{lang}")
    return queries


def _topic_queries(topics: list[str]) -> list[str]:
    """Star-sorted topic queries that surface flagship repos keyword search misses.

    GitHub topics are curated metadata, so ``topic:message-broker sort:stars-desc``
    reliably brings up ``apache/kafka``/``nats`` even though their descriptions omit
    the words a plain keyword search would need.
    """
    seen: set[str] = set()
    out: list[str] = []
    for slug in topics[:5]:
        slug = slug.strip().lower().replace(" ", "-")
        if slug and slug not in seen:
            seen.add(slug)
            out.append(f"topic:{slug} sort:stars-desc")
    return out


def _card(index: int, rec: RepoRecord) -> str:
    """Compact, token-cheap evidence card — signals + README excerpt, no code."""
    readme = (rec.readme or rec.description or "").strip()
    readme = " ".join(readme.split())[:700]
    topics = ", ".join(rec.topics[:8])
    return (
        f"[{index}] {rec.name_with_owner}\n"
        f"    signals: {signal_summary(rec)}\n"
        f"    topics: {topics or '(none)'}\n"
        f"    readme: {readme or '(none)'}"
    )


def _format_report(
    intent: str,
    ranked: list[RankedRepo],
    summary: str,
    trace: list[str] | None = None,
) -> str:
    lines = ["# DeepGit results", "", f"**Intent:** {intent}", ""]
    if summary:
        lines += [f"> {summary}", ""]
    medals = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(ranked):
        medal = medals[i] if i < len(medals) else f"{i + 1}."
        lines.append(
            f"## {medal} {r.repo}  —  {r.verdict} ({r.fit_score:.2f})"
        )
        lines.append(f"https://github.com/{r.repo}")
        if r.why:
            lines.append(f"\n{r.why}")
        if r.risks:
            lines.append("\n**Risks:** " + "; ".join(r.risks))
        lines.append("")
    if trace:
        lines.append("---")
        lines.append("### How DeepGit decided")
        for step in trace:
            lines.append(f"- {step}")
        lines.append("")
    return "\n".join(lines)


def _ranking_for_review(ranked: list[RankedRepo], evidence: str) -> str:
    """Render the current ranking + verified code evidence for the reflector."""
    lines = ["CURRENT RANKING:"]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"{i}. {r.repo} — {r.verdict} (fit {r.fit_score:.2f}) — {r.why}"
        )
        if r.risks:
            lines.append(f"   risks: {'; '.join(r.risks)}")
    if evidence:
        lines.append("\nVERIFIED CODE EVIDENCE:")
        lines.append(evidence)
    return "\n".join(lines)


def _apply_critique(ranked: list[RankedRepo], critique) -> list[RankedRepo]:
    """Promote the critic's chosen repo to #1 when it overrides with confidence."""
    if critique.approved or not critique.top_pick or critique.confidence < 0.6:
        return ranked
    idx = next(
        (i for i, r in enumerate(ranked) if r.repo == critique.top_pick), None
    )
    if idx is None or idx == 0:
        return ranked
    promoted = ranked.pop(idx)
    ranked.insert(0, promoted)
    logger.info("[pipeline] reflection promoted %s to #1", critique.top_pick)
    return ranked


def search(
    intent: str,
    *,
    candidates: int = 40,
    shortlist: int = 8,
    per_query: int = 15,
    adaptive: bool = True,
) -> str:
    """Find the best repositories for ``intent`` — fast, cheap, grounded.

    Two LLM calls in the common case. When ``adaptive`` is on (default) and the
    fast ranking is genuinely uncertain, DeepGit escalates: it reads the *real
    code* of only the contested repos, re-judges from that code, and a reflection
    critic reviews the ranking before returning. Effort matches uncertainty.
    Returns a Markdown report.
    """
    result = search_structured(
        intent,
        candidates=candidates,
        shortlist=shortlist,
        per_query=per_query,
        adaptive=adaptive,
    )
    if not result.ranked and not result.candidates_scanned:
        return f"# DeepGit results\n\nNo repositories found for: {intent}"
    return _format_report(intent, result.ranked, result.summary, result.trace)


def search_structured(
    intent: str,
    *,
    candidates: int = 40,
    shortlist: int = 8,
    per_query: int = 15,
    adaptive: bool = True,
) -> SearchResult:
    """Core search returning a structured :class:`SearchResult`.

    This is the reusable engine behind :func:`search` (which just formats the
    Markdown). Benchmarks, the CLI, and the MCP server call this so they get the
    ranked list, the reasoning trace, and telemetry (LLM-call count, escalation
    flag, timing) without re-parsing a report.
    """
    t0 = time.time()
    configure_dspy()
    trace: list[str] = []
    llm_calls = 0

    # 1) Understand the intent (1 LLM call).
    plan = IntentExpander()(intent=intent)
    llm_calls += 1
    keyword_queries = plan.search_queries or [intent]
    topic_queries = _topic_queries(getattr(plan, "github_topics", []) or [])
    queries = keyword_queries + topic_queries
    logger.info(
        "[pipeline] plan: %d keyword + %d topic queries, must_have=%s",
        len(keyword_queries), len(topic_queries), plan.must_have,
    )

    # 2) Gather candidates (batched GraphQL, 0 LLM tokens). Keyword queries find
    #    literal matches; topic queries (sort:stars-desc) surface flagship repos
    #    whose descriptions omit generic terms (e.g. apache/kafka).
    recs = asyncio.run(_gather(queries, per_query))
    logger.info("[pipeline] gathered %d unique candidates", len(recs))

    # Safety net: if the LLM's queries were too narrow, retry with short
    # keyword queries derived from the intent itself (still 0 LLM tokens).
    if len(recs) < 5:
        terms_list = sorted(intent_terms(intent))
        fallback = _fallback_queries(terms_list, plan.ecosystems)
        if fallback:
            logger.info("[pipeline] sparse results, broad retry: %s", fallback)
            extra = asyncio.run(_gather(fallback, per_query))
            by_name = {r.name_with_owner: r for r in recs}
            for r in extra:
                by_name.setdefault(r.name_with_owner, r)
            recs = list(by_name.values())
            logger.info("[pipeline] after retry: %d candidates", len(recs))

    # 2b) Semantic layer: persist everything we gathered into the local vector
    #     index, then pull in semantically-near repos from the accumulated corpus
    #     that keyword+topic search did not surface this time. This is DeepGit's
    #     cross-query memory — recall compounds the more it is used. No-op when the
    #     optional embedding deps are absent.
    index = get_index()
    if index.available and recs:
        stored = index.upsert(recs)
        have = {r.name_with_owner for r in recs}
        sem = index.search(intent, k=15, exclude=have)
        if sem:
            recs.extend(sem)
            logger.info(
                "[pipeline] semantic recall: +%d repos from corpus (%d indexed): %s",
                len(sem), index.count(),
                ", ".join(r.name_with_owner for r in sem[:6]),
            )
            trace.append(
                f"Semantic recall added **{len(sem)}** repo(s) from the local vector "
                f"index that keyword search missed."
            )
        else:
            logger.debug("[pipeline] semantic recall: 0 new (indexed %d)", stored)

    if not recs:
        return SearchResult(
            intent=intent, ranked=[], summary="", trace=["No repositories found."],
            candidates_scanned=0, escalated=False, llm_calls=llm_calls,
            elapsed=time.time() - t0,
        )

    trace.append(f"Scanned **{len(recs)}** repositories from {len(queries)} search angles.")

    # 3) Zero-token signal prefilter -> shortlist.
    terms = intent_terms(intent)
    recs.sort(key=lambda r: prescore(terms, r), reverse=True)
    top = recs[: min(shortlist, candidates)]
    records = {r.name_with_owner: r for r in top}
    logger.info(
        "[pipeline] shortlist: %s",
        ", ".join(r.name_with_owner for r in top),
    )
    trace.append(
        f"Pre-ranked on free signals (activity, health, relevance) and shortlisted "
        f"the top **{len(top)}** for judgment."
    )

    # 4) One batched LLM judgment over compact evidence cards.
    cards = "\n\n".join(_card(i + 1, r) for i, r in enumerate(top))
    pred = RepoRanker()(
        intent=intent,
        candidates=cards,
        must_have=plan.must_have,
        anti_patterns=plan.anti_patterns,
        target_languages=plan.ecosystems,
    )
    llm_calls += 1
    ranked = list(pred.ranked or [])
    ranked.sort(key=lambda r: r.fit_score, reverse=True)
    trace.append("Judged the shortlist from README + signals (1 LLM call).")

    escalated = False
    # 5) Adaptive controller — the agentic core. Zero-token confidence gate
    #    decides whether the fast answer is trustworthy; only if not do we read
    #    real code for the contested repos and reflect on the result.
    if adaptive and ranked:
        confidence = assess_confidence(ranked)
        logger.info("[pipeline] confidence: %s", confidence.summary)
        if not confidence.escalate:
            trace.append(f"Confidence gate: **{confidence.summary}**.")
        else:
            escalated = True
            trace.append(
                f"Confidence gate: **{confidence.summary}** → verifying "
                f"{', '.join(confidence.contested)} from real code."
            )
            esc = verify_contested(
                confidence, records, intent, plan.must_have, plan.anti_patterns
            )
            llm_calls += len(confidence.contested)
            ranked = apply_escalation(ranked, esc)
            for note in esc.notes:
                trace.append(f"Code check — {note}")

            # Reflection: a skeptical critic reviews the corrected ranking.
            ranking_text = _ranking_for_review(ranked, esc.evidence)
            critique = RankingReflector()(
                intent=intent,
                ranking=ranking_text,
                must_have=plan.must_have,
                anti_patterns=plan.anti_patterns,
                target_languages=plan.ecosystems,
            )
            llm_calls += 1
            ranked = _apply_critique(ranked, critique)
            verdict = "approved current #1" if critique.approved else (
                f"promoted {critique.top_pick} to #1"
            )
            trace.append(
                f"Reflection critic ({critique.confidence:.2f}): {verdict}"
                + (f" — {critique.note}" if critique.note else "")
            )

    elapsed = time.time() - t0
    logger.info("[pipeline] done in %.1fs (%d LLM calls)", elapsed, llm_calls)
    trace.append(f"Completed in {elapsed:.1f}s.")
    return SearchResult(
        intent=intent,
        ranked=ranked,
        summary=pred.summary,
        trace=trace,
        candidates_scanned=len(recs),
        escalated=escalated,
        llm_calls=llm_calls,
        elapsed=elapsed,
    )
