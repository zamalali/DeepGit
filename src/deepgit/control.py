from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from deepgit.github.graphql import GitHubGraphQL
from deepgit.github.schema import RepoRecord, TreeEntry
from deepgit.reasoning import FitJudge, RankedRepo, RankingReflector

logger = logging.getLogger("deepgit.control")

# Escalation is warranted when the top of the ranking is not clearly decided.
# A *genuine* dead heat means #1 and #2 are effectively tied (margin at or below
# this, after rounding away float noise) AND share the same verdict. The ranker
# emits fit-scores on a compressed ~0.05 grid, so this is deliberately tight: two
# "strong" repos a notch apart is a confident answer, not an uncertain one.
# Escalating on every multi-strong shortlist (the original bug) spends effort
# where there is no real uncertainty — and the benchmark showed it can even
# regress a correct canonical pick.
TIE_MARGIN = 0.04  # rounded margin at/below this AND same verdict => dead heat
# When escalation *is* triggered, a challenger only unseats the fast lane's #1 if
# its code-verified score beats the incumbent's by this much. Equal-strength repos
# must not reshuffle the top — that adds noise, not signal.
DECISIVE_MARGIN = 0.10
MAX_VERIFY = 3  # never read code for more than this many repos
MAX_FILES_PER_REPO = 4  # bound the evidence per repo
MAX_FILE_BYTES = 8_000  # truncate each file — we judge design, not every line

# Directories that usually hold the real implementation, in priority order.
_SOURCE_DIRS = ("src", "lib", "source")
# Directories that hold tests.
_TEST_DIRS = ("tests", "test", "spec", "__tests__")
# Root files worth reading directly as capability evidence.
_ROOT_CODE = ("pyproject.toml", "setup.py", "package.json", "cargo.toml", "go.mod")
# Blob extensions we consider "code" worth reading.
_CODE_EXT = (
    ".py", ".ts", ".tsx", ".js", ".rs", ".go", ".java", ".rb", ".cpp", ".c",
    ".h", ".hpp", ".cs", ".kt", ".swift", ".scala",
)


@dataclass
class Confidence:
    """The controller's verdict on whether the fast ranking can be trusted."""

    escalate: bool = False
    reasons: list[str] = field(default_factory=list)
    contested: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.escalate:
            return "confident — clear winner, no code verification needed"
        return "uncertain — " + "; ".join(self.reasons)


def assess_confidence(ranked: list[RankedRepo]) -> Confidence:
    """Zero-token gate: decide if the fast ranking is trustworthy as-is.

    The gate escalates only when the **top pick itself** is in genuine doubt —
    not merely because the shortlist happens to contain several good repos. A
    confident answer is a strong #1 that the ranker separates from #2; reading
    code in that case just burns effort for no gain (the benchmark showed blanket
    escalation can even *regress* a correct canonical pick, and fired on ~100% of
    queries because top fit-scores cluster tightly).

    Uncertainty signals (any one triggers verification):
      * the #1 verdict is not "strong" (the ranker itself is not confident);
      * a genuine dead heat at the top — #1 and #2 effectively tied (rounded
        margin at or below ``TIE_MARGIN``) *and* sharing the same verdict.

    Note: risks on an already-"strong" #1 do **not** trigger escalation — the
    judge weighed those risks and still called it strong. They stay informational.
    """
    if not ranked:
        return Confidence(escalate=False, reasons=["no candidates"], contested=[])

    top = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    reasons: list[str] = []
    contested: list[str] = [top.repo]

    # Primary trigger: the ranker is not confident the #1 is a strong fit.
    if top.verdict != "strong":
        reasons.append(f"#1 verdict is '{top.verdict}', not 'strong'")

    # Tie trigger: a real dead heat, not just "#2 is also good". Round the margin
    # to kill float noise (0.90-0.80 is 0.0999… in binary float).
    tie = False
    if runner is not None:
        margin = round(top.fit_score - runner.fit_score, 2)
        if margin <= TIE_MARGIN and runner.verdict == top.verdict:
            tie = True
            reasons.append(
                f"dead heat: margin {margin:.2f} between #1 ({top.repo}) and "
                f"#2 ({runner.repo}), both '{top.verdict}'"
            )

    escalate = bool(reasons)

    # Only once we've decided to escalate do we widen the contested set to the
    # other near-top repos that could plausibly deserve #1. Widening never *causes*
    # escalation on its own — that was the over-firing bug the benchmark exposed.
    if escalate:
        if tie and runner is not None:
            contested.append(runner.repo)
        for r in ranked[1:MAX_VERIFY]:
            if r.verdict == "strong" and r.repo not in contested:
                contested.append(r.repo)

    return Confidence(
        escalate=escalate,
        reasons=reasons,
        contested=contested[:MAX_VERIFY],
    )


def _pick_root_paths(rec: RepoRecord) -> list[str]:
    """Manifest/entry files at the repo root worth reading as evidence."""
    names = {e.name.lower(): e.name for e in rec.root_tree if e.type == "blob"}
    return [names[k] for k in _ROOT_CODE if k in names]


async def _pick_source_paths(
    gh: GitHubGraphQL, rec: RepoRecord, budget: int
) -> list[str]:
    """Look one level into the likely source dir and pick real code files."""
    if budget <= 0:
        return []
    dir_names = {e.name.lower(): e for e in rec.root_tree if e.type == "tree"}

    # Prefer a conventional source dir, else a dir named like the package/repo.
    target: TreeEntry | None = None
    for cand in _SOURCE_DIRS:
        if cand in dir_names:
            target = dir_names[cand]
            break
    if target is None:
        pkg = rec.name.replace("-", "_").lower()
        for low, entry in dir_names.items():
            if low == pkg or low == rec.name.lower():
                target = entry
                break

    paths: list[str] = []
    if target is not None:
        entries = await gh.fetch_tree(
            rec.name_with_owner, target.path, ref=rec.default_branch
        )
        # Prefer an __init__/index/mod entry, then the largest code file.
        blobs = [e for e in entries if e.type == "blob" and e.name.endswith(_CODE_EXT)]
        blobs.sort(key=lambda e: e.size, reverse=True)
        entrypoints = [
            e for e in blobs
            if e.name.lower().split(".")[0] in ("__init__", "index", "mod", "lib", "main")
        ]
        ordered = entrypoints + [b for b in blobs if b not in entrypoints]
        paths = [e.path for e in ordered[:budget]]

    # Fallback: a top-level code file at the root.
    if not paths:
        root_code = [
            e.name for e in rec.root_tree
            if e.type == "blob" and e.name.endswith(_CODE_EXT)
        ]
        paths = root_code[:budget]
    return paths


async def _pick_test_path(gh: GitHubGraphQL, rec: RepoRecord) -> list[str]:
    """Find one real test file, proving the project actually tests itself."""
    dir_names = {e.name.lower(): e for e in rec.root_tree if e.type == "tree"}
    for cand in _TEST_DIRS:
        if cand in dir_names:
            entries = await gh.fetch_tree(
                rec.name_with_owner, dir_names[cand].path, ref=rec.default_branch
            )
            tests = [
                e for e in entries
                if e.type == "blob" and e.name.endswith(_CODE_EXT)
                and ("test" in e.name.lower() or "spec" in e.name.lower())
            ]
            if tests:
                return [tests[0].path]
    return []


async def _gather_code_evidence(
    rec: RepoRecord,
) -> dict[str, str]:
    """Read the *real* source of one repo: manifests + a source file + a test.

    This is the honest "reads code" step — bounded, targeted, one repo at a time.
    """
    async with GitHubGraphQL() as gh:
        root = _pick_root_paths(rec)
        remaining = MAX_FILES_PER_REPO - len(root)
        source = await _pick_source_paths(gh, rec, max(0, remaining - 1))
        test = await _pick_test_path(gh, rec)
        wanted = (root + source + test)[:MAX_FILES_PER_REPO]
        if not wanted:
            return {}
        logger.info("[control] reading %s <- %s", rec.name_with_owner, wanted)
        return await gh.fetch_files(
            rec.name_with_owner, wanted, ref=rec.default_branch, max_bytes=MAX_FILE_BYTES
        )


def _code_evidence_block(rec: RepoRecord, files: dict[str, str]) -> str:
    """Assemble the metadata + README + real code excerpts for one repo."""
    from deepgit.retrieval import signal_summary

    parts = [
        f"REPO: {rec.name_with_owner}",
        f"signals: {signal_summary(rec)}",
        f"topics: {', '.join(rec.topics[:10]) or '(none)'}",
        "",
        "README (excerpt):",
        (rec.readme or rec.description or "(none)")[:2500],
    ]
    if files:
        parts.append("\nREAL CODE (excerpts):")
        for path, text in files.items():
            parts.append(f"\n--- {path} ---\n{text[:MAX_FILE_BYTES]}")
    return "\n".join(parts)


@dataclass
class Escalation:
    """The outcome of a deep-verification pass over the contested repos."""

    verified: dict[str, str] = field(default_factory=dict)  # repo -> verdict
    scores: dict[str, float] = field(default_factory=dict)  # repo -> code-grounded fit
    notes: list[str] = field(default_factory=list)
    evidence: str = ""  # combined evidence block, fed to the reflector


def verify_contested(
    confidence: Confidence,
    records: dict[str, RepoRecord],
    intent: str,
    must_have: list[str],
    anti_patterns: list[str],
) -> Escalation:
    """Read real code for the contested repos and re-judge from that code.

    One ``JudgeFit`` call per contested repo, grounded in actual source — this is
    where DeepGit earns the "reads the code" claim, but only for the 2-3 repos
    whose ranking is genuinely in doubt.
    """
    judge = FitJudge()
    esc = Escalation()
    evidence_blocks: list[str] = []

    for name in confidence.contested:
        rec = records.get(name)
        if rec is None:
            continue
        files = asyncio.run(_gather_code_evidence(rec))
        block = _code_evidence_block(rec, files)
        evidence_blocks.append(block)

        verdict = judge(
            intent=intent,
            evidence=block,
            must_have=must_have,
            anti_patterns=anti_patterns,
        )
        esc.verified[name] = verdict.verdict
        esc.scores[name] = float(verdict.fit_score)
        note = (
            f"{name}: code-verified -> {verdict.verdict} "
            f"(fit {verdict.fit_score:.2f})"
        )
        if verdict.risks:
            note += " · risks: " + "; ".join(verdict.risks[:2])
        esc.notes.append(note)
        logger.info("[control] %s", note)

    esc.evidence = "\n\n".join(evidence_blocks)
    return esc


def apply_escalation(
    ranked: list[RankedRepo], esc: Escalation
) -> list[RankedRepo]:
    """Fold code-grounded verdicts back into the ranking and re-sort.

    Escalation *corrects* the ranking when reading real code reveals a genuine
    difference — but it should not reshuffle repos that come out equally strong.
    So the fast lane's #1 is treated as the incumbent: a challenger only takes the
    top spot if its code-verified score beats the incumbent's by ``DECISIVE_MARGIN``.
    A dead-heat after code reading keeps the incumbent, which is the stable, honest
    default (the benchmark showed unanchored re-sorting can demote a correct pick).
    """
    if not ranked:
        return ranked
    incumbent = ranked[0].repo
    for r in ranked:
        if r.repo in esc.scores:
            r.fit_score = esc.scores[r.repo]
            r.verdict = esc.verified.get(r.repo, r.verdict)  # type: ignore[assignment]
    inc = next((r for r in ranked if r.repo == incumbent), None)
    ranked.sort(key=lambda r: r.fit_score, reverse=True)
    # Incumbency anchor: only a decisive code-verified lead unseats the fast #1.
    if inc is not None and ranked[0].repo != incumbent:
        if ranked[0].fit_score - inc.fit_score < DECISIVE_MARGIN:
            ranked.remove(inc)
            ranked.insert(0, inc)
    return ranked
