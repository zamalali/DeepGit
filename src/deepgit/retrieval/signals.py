from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from deepgit.github.schema import RepoRecord

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common words that carry no discriminative signal for repo matching.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "with", "that",
    "is", "it", "as", "at", "by", "be", "this", "i", "need", "want", "looking",
    "library", "tool", "package", "framework", "python", "not", "must", "should",
    "actively", "maintained", "lightweight", "simple", "easy", "best", "good",
}


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def intent_terms(intent: str) -> set[str]:
    """Discriminative terms from the intent, stopwords removed."""
    return {t for t in _tokens(intent) if t not in _STOPWORDS and len(t) > 1}


def days_since(iso: str) -> int:
    if not iso:
        return 9999
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return max(0, (datetime.now(timezone.utc) - dt).days)
    except ValueError:
        return 9999


def maintenance_score(rec: RepoRecord) -> float:
    """1.0 = pushed very recently, decaying to ~0 around 3 years stale."""
    if rec.is_archived:
        return 0.0
    d = days_since(rec.pushed_at)
    return max(0.0, min(1.0, 1.0 - math.log1p(d) / math.log1p(1095)))


def popularity_score(rec: RepoRecord) -> float:
    """Log-scaled stars; ~10k stars saturates to 1.0."""
    return max(0.0, min(1.0, math.log1p(rec.stars) / math.log1p(10000)))


def health_score(rec: RepoRecord) -> float:
    """Project hygiene from the file tree: tests, CI, license, manifest."""
    s = 0.0
    if rec.has_tests():
        s += 0.40
    if rec.has_ci():
        s += 0.25
    if rec.license_spdx and rec.license_spdx != "NOASSERTION":
        s += 0.20
    if rec.manifest_files():
        s += 0.15
    return min(1.0, s)


def relevance_score(terms: set[str], rec: RepoRecord) -> float:
    """Fraction of intent terms found in the repo's text surface."""
    if not terms:
        return 0.0
    hay = (
        _tokens(rec.name)
        | _tokens(rec.description)
        | _tokens(" ".join(rec.topics))
        | _tokens(rec.readme[:2000])
    )
    return min(1.0, len(terms & hay) / len(terms))


# Prefilter fusion weights. This is a *prefilter only* — its single job is to make
# sure the right candidates reach the LLM shortlist, NOT to be the final ranking.
# Relevance still leads, but popularity is weighted meaningfully because a keyword
# prefilter otherwise lets tiny keyword-stuffed clones (e.g. "rust-async-web-server",
# 3 stars) crowd out the canonical high-signal repos (axum, actix-web) that a
# developer actually wants. The LLM then judges the shortlist on merit, so this
# lean toward popularity does NOT make the final answer star-biased.
W_RELEVANCE = 0.40
W_MAINTENANCE = 0.20
W_HEALTH = 0.10
W_POPULARITY = 0.30


def prescore(terms: set[str], rec: RepoRecord) -> float:
    """Cheap 0..1 prefilter score. Used only to pick the LLM shortlist."""
    return (
        W_RELEVANCE * relevance_score(terms, rec)
        + W_MAINTENANCE * maintenance_score(rec)
        + W_HEALTH * health_score(rec)
        + W_POPULARITY * popularity_score(rec)
    )


def signal_summary(rec: RepoRecord) -> str:
    """One-line human/LLM-readable signal digest (no code, cheap)."""
    d = days_since(rec.pushed_at)
    freshness = (
        "archived" if rec.is_archived
        else f"pushed {d}d ago" if d < 9999
        else "unknown activity"
    )
    lic = rec.license_spdx or "no-license"
    bits = [
        f"{rec.stars}*",
        rec.primary_language or "?",
        lic,
        freshness,
        f"tests={'y' if rec.has_tests() else 'n'}",
        f"ci={'y' if rec.has_ci() else 'n'}",
    ]
    manifests = rec.manifest_files()
    if manifests:
        bits.append("manifest=" + ",".join(manifests[:3]))
    return " · ".join(bits)
