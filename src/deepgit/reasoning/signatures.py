from __future__ import annotations

from typing import Literal

import dspy
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------
class IntentPlan(BaseModel):
    """A search plan derived from a natural-language intent.

    This is the antidote to the old ``query -> keyword tags`` step: instead of
    collapsing intent into a handful of lossy keywords, we keep the full intent
    and express it as several complementary search angles plus explicit
    hard/soft constraints the judge can check against.
    """

    search_queries: list[str] = Field(
        default_factory=list,
        description="3-6 GitHub search queries. Each MUST be SHORT — 1 to 3 core "
        "keywords only (GitHub ANDs every term, so long queries match nothing). "
        "Prefer the domain nouns a repo would actually contain, e.g. 'icalendar "
        "python', 'ics parser', 'calendar rfc5545'. You may append ONE qualifier "
        "like 'language:python'. Do NOT put adjectives like 'lightweight', "
        "'best', or 'actively maintained' in the query — those are judged later.",
    )
    ecosystems: list[str] = Field(
        default_factory=list,
        description="Languages/frameworks to prioritise, e.g. ['python', 'rust']. "
        "When the intent NAMES a language ('a c++ library', 'in python', 'for "
        "go'), put that language FIRST — it is a hard requirement, not a "
        "preference: a repo written in a different language does not satisfy the "
        "need unless it ships first-class bindings for the named language.",
    )
    github_topics: list[str] = Field(
        default_factory=list,
        description="2-5 GitHub TOPIC slugs the canonical repos are tagged with, "
        "lowercase-hyphenated, e.g. ['message-broker', 'pubsub', 'messaging'] or "
        "['orm', 'database']. These are searched as `topic:<slug> sort:stars-desc` "
        "to surface flagship projects (kafka, nats) that plain keyword search "
        "misses because their descriptions omit generic terms. Prefer real, "
        "widely-used topic slugs over invented ones.",
    )
    must_have: list[str] = Field(
        default_factory=list,
        description="Hard capability requirements a good result MUST satisfy.",
    )
    nice_to_have: list[str] = Field(
        default_factory=list,
        description="Soft preferences that improve fit but are not required.",
    )
    anti_patterns: list[str] = Field(
        default_factory=list,
        description="Things to avoid, e.g. 'abandoned', 'thin wrapper around X', "
        "'not the framework the user already rejected'. When the intent asks for "
        "a general-purpose tool and does NOT mention specialised hardware, add "
        "'requires specialised hardware (e.g. GPU-only)' — a GPU-bound library is "
        "a poor default for a plain 'fast X' request.",
    )
    rationale: str = Field(
        default="", description="Brief explanation of the strategy."
    )


class RepoVerdict(BaseModel):
    """A calibrated fit judgment for one repository."""

    fit_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall fit, 0..1."
    )
    verdict: Literal["strong", "partial", "weak", "off_target"] = Field(
        default="weak"
    )
    satisfied_must_haves: list[str] = Field(default_factory=list)
    missing_must_haves: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(
        default_factory=list,
        description="Evidence-cited reasons for the score (reference the code/README).",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Maintenance, licensing, maturity, or fit risks.",
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in this judgment."
    )


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------
class ExpandIntent(dspy.Signature):
    """Turn a developer's natural-language need into a concrete search plan.

    Preserve the full intent. Do NOT reduce it to a few keywords. Capture the
    hard requirements, the soft preferences, and the anti-patterns explicitly so
    a later judge can verify each candidate against them. Also name the GitHub
    topic slugs the flagship repos are tagged with, so star-sorted topic search
    can surface canonical projects that keyword search alone would miss.
    """

    intent: str = dspy.InputField(desc="What the developer is looking for.")
    plan: IntentPlan = dspy.OutputField(desc="Structured search plan.")


class JudgeFit(dspy.Signature):
    """Judge how well a repository fits the intent, grounded in real evidence.

    You are given the intent's constraints and concrete evidence about the repo
    (metadata, README, and excerpts of actual code/tests/manifests). Judge only
    from the evidence. Reward repos that demonstrably satisfy the must-haves;
    penalise unmaintained, mismatched, or superficial matches. Cite the evidence
    in your reasons.
    """

    intent: str = dspy.InputField(desc="The developer's need.")
    must_have: list[str] = dspy.InputField(desc="Hard requirements.")
    anti_patterns: list[str] = dspy.InputField(desc="Things to avoid.")
    evidence: str = dspy.InputField(
        desc="Metadata + README + code/test/manifest excerpts for the repo."
    )
    verdict: RepoVerdict = dspy.OutputField(desc="Calibrated fit judgment.")


class SynthesizeReport(dspy.Signature):
    """Produce a ranked, decision-ready recommendation from judged repositories.

    Rank by genuine fit to the intent (not stars). Surface hidden gems that
    strongly match even if less popular. Be concise and concrete; explain WHY
    the top pick wins and when an alternative is the better choice.
    """

    intent: str = dspy.InputField(desc="The developer's need.")
    judged_repos: str = dspy.InputField(
        desc="Repositories with their verdicts, scores, reasons, and risks."
    )
    report: str = dspy.OutputField(desc="Ranked recommendation in Markdown.")
    top_pick: str = dspy.OutputField(desc="full_name of the single best repo.")


class RankedRepo(BaseModel):
    """One entry in a ranked recommendation."""

    repo: str = Field(default="", description="owner/name of the repository.")
    fit_score: float = Field(default=0.0, ge=0.0, le=1.0)
    verdict: Literal["strong", "partial", "weak", "off_target"] = Field(default="weak")
    why: str = Field(default="", description="One-sentence reason this repo fits (or not).")
    risks: list[str] = Field(default_factory=list)


class RankRepositories(dspy.Signature):
    """Rank a shortlist of candidate repositories by true fit to the intent.

    You are given the intent's hard requirements and anti-patterns, plus a
    compact evidence card for each candidate: metadata signals (stars, activity,
    license, tests/CI) and a README excerpt describing what it does. Judge from
    that evidence — you do NOT need the full source code; the README, the
    activity signals, and the project hygiene are enough to decide fit reliably.

    Reward repos that clearly do what the intent needs and are healthy/maintained.
    Penalise stale, archived, superficial, or off-target repos. Do not just rank
    by stars — a smaller, well-matched repo can win. Return every candidate,
    ordered best-first.

    HARD CONSTRAINTS. When ``target_languages`` is non-empty, the intent names a
    required language: a candidate whose primary language differs — and which
    does not expose first-class bindings for that language — is at best a ``weak``
    match and must never be #1 (e.g. a C JSON parser cannot win a request for a
    C++ JSON library). Likewise, when an anti-pattern flags specialised hardware,
    a library that only runs on that hardware (e.g. a GPU-only dataframe) must not
    outrank an equally-capable general-purpose one for a plain 'fast X' request.
    """

    intent: str = dspy.InputField(desc="The developer's need.")
    must_have: list[str] = dspy.InputField(desc="Hard requirements.")
    anti_patterns: list[str] = dspy.InputField(desc="Things to avoid.")
    target_languages: list[str] = dspy.InputField(
        desc="Required language(s) when the intent names one; empty if none. A "
        "candidate in a different language (without first-class bindings) fails "
        "this hard constraint."
    )
    candidates: str = dspy.InputField(
        desc="Numbered evidence cards: signals + README excerpt per repo."
    )
    ranked: list[RankedRepo] = dspy.OutputField(
        desc="All candidates, best-first, with fit_score, verdict, why, risks."
    )
    summary: str = dspy.OutputField(
        desc="2-3 sentence bottom line: the top pick and when an alternative wins."
    )


class RankingCritique(BaseModel):
    """A critic's review of a proposed ranking before it is returned."""

    approved: bool = Field(
        default=True,
        description="True if the current #1 pick is genuinely the best choice.",
    )
    top_pick: str = Field(
        default="",
        description="owner/name that SHOULD be #1 after review (may differ from current).",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Specific, evidence-cited problems with the current ranking.",
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence in this critique."
    )
    note: str = Field(
        default="", description="One-line explanation of the verdict."
    )


class ReflectOnRanking(dspy.Signature):
    """Critically review a proposed ranking against the evidence before returning it.

    You are a skeptical senior engineer doing a final sanity check. You are given
    the intent's hard requirements and anti-patterns, and the current ranking with
    each repo's reasons, risks, and (for verified repos) real code/manifest
    excerpts. Look for mistakes the ranker may have made: a thin wrapper ranked
    above the library it wraps, a popular-but-mismatched repo ranked on stars, a
    stale/abandoned pick, an anti-pattern that was missed, or a genuinely better
    repo sitting at #2. Decide whether the #1 pick is truly the best; if not, name
    the repo that should be #1. Only override when the evidence clearly supports it.
    """

    intent: str = dspy.InputField(desc="The developer's need.")
    must_have: list[str] = dspy.InputField(desc="Hard requirements.")
    anti_patterns: list[str] = dspy.InputField(desc="Things to avoid.")
    target_languages: list[str] = dspy.InputField(
        desc="Required language(s) when the intent names one; empty if none. The "
        "#1 pick must satisfy this — a repo in a different language (without "
        "first-class bindings) should not stand as #1."
    )
    ranking: str = dspy.InputField(
        desc="Current ranking with reasons, risks, and code/manifest evidence."
    )
    critique: RankingCritique = dspy.OutputField(
        desc="Review verdict: approved, corrected top_pick, concerns, confidence."
    )



# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class IntentExpander(dspy.Module):
    """Reasoning module for intent -> search plan."""

    def __init__(self) -> None:
        super().__init__()
        self.expand = dspy.ChainOfThought(ExpandIntent)

    def forward(self, intent: str) -> IntentPlan:
        return self.expand(intent=intent).plan


class FitJudge(dspy.Module):
    """Reasoning module for evidence-grounded fit judgment."""

    def __init__(self) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(JudgeFit)

    def forward(
        self,
        intent: str,
        evidence: str,
        must_have: list[str] | None = None,
        anti_patterns: list[str] | None = None,
    ) -> RepoVerdict:
        return self.judge(
            intent=intent,
            must_have=must_have or [],
            anti_patterns=anti_patterns or [],
            evidence=evidence,
        ).verdict


class ReportSynthesizer(dspy.Module):
    """Reasoning module for the final ranked report."""

    def __init__(self) -> None:
        super().__init__()
        self.synth = dspy.ChainOfThought(SynthesizeReport)

    def forward(self, intent: str, judged_repos: str) -> dspy.Prediction:
        return self.synth(intent=intent, judged_repos=judged_repos)


class RepoRanker(dspy.Module):
    """One-shot batched ranker — the fast lane's single judgment call."""

    def __init__(self) -> None:
        super().__init__()
        self.rank = dspy.ChainOfThought(RankRepositories)

    def forward(
        self,
        intent: str,
        candidates: str,
        must_have: list[str] | None = None,
        anti_patterns: list[str] | None = None,
        target_languages: list[str] | None = None,
    ) -> dspy.Prediction:
        return self.rank(
            intent=intent,
            must_have=must_have or [],
            anti_patterns=anti_patterns or [],
            target_languages=target_languages or [],
            candidates=candidates,
        )


class RankingReflector(dspy.Module):
    """Self-critique module — reviews a ranking against evidence before return.

    This is the reflection stage of the adaptive controller: a cheap final guard
    that catches ranking mistakes (thin wrappers, stars-over-fit, missed
    anti-patterns) and can promote a better repo to #1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reflect = dspy.ChainOfThought(ReflectOnRanking)

    def forward(
        self,
        intent: str,
        ranking: str,
        must_have: list[str] | None = None,
        anti_patterns: list[str] | None = None,
        target_languages: list[str] | None = None,
    ) -> RankingCritique:
        return self.reflect(
            intent=intent,
            must_have=must_have or [],
            anti_patterns=anti_patterns or [],
            target_languages=target_languages or [],
            ranking=ranking,
        ).critique
