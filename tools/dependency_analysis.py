# -*- coding: utf-8 -*-
import base64
import httpx
import logging
import os
import toml
from functools import lru_cache
from tools.chat import chain

logger = logging.getLogger(__name__)

QUESTION_TMPL = (
    "Given the following Python dependencies, can this project run on {hw}? "
    "Answer YES or NO and one short reason.\n\nDependencies:\n{deps}"
)

# --------------------------------------------------------------------------- #
# GitHub helper – cached raw file fetch
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2048)
def _gh_raw(owner: str, repo: str, path: str, token: str) -> str | None:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}"} if token else {}
    r = httpx.get(url, headers=headers)
    if r.status_code != 200:
        return None
    data = r.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8")
    return data.get("content", "")

def _collect_deps(owner: str, repo: str, token: str) -> list[str]:
    """Return a flat list of dependency strings (possibly empty)."""
    deps: list[str] = []

    # requirements.txt
    req = _gh_raw(owner, repo, "requirements.txt", token) or ""
    deps += [ln.strip() for ln in req.splitlines() if ln.strip() and not ln.startswith("#")]

    # pyproject.toml (Poetry style)
    py = _gh_raw(owner, repo, "pyproject.toml", token) or ""
    if py:
        try:
            deps += list(
                toml.loads(py).get("tool", {})
                               .get("poetry", {})
                               .get("dependencies", {})
                               .keys()
            )
        except Exception:
            pass
    return deps

# --------------------------------------------------------------------------- #
# LangGraph node
# --------------------------------------------------------------------------- #
def dependency_analysis(state, config):
    hardware = state.hardware_spec       # None  → no filtering
    candidates = state.filtered_candidates

    if not hardware:
        state.hardware_filtered = candidates
        logger.info("[Deps] no hardware constraint – skipping check")
        return {"hardware_filtered": candidates}

    token = os.getenv("GITHUB_API_KEY", "")
    kept, dropped = [], []

    for repo in candidates:
        full = repo.get("full_name", "")
        if "/" not in full:
            kept.append(repo)
            continue

        owner, name = full.split("/", 1)
        deps = _collect_deps(owner, name, token)

        if not deps:                      # assume lightweight if no deps
            kept.append(repo)
            continue

        prompt = QUESTION_TMPL.format(hw=hardware, deps=", ".join(deps[:30]))
        answer = chain.invoke({"query": prompt}).content.strip().lower()
        verdict = answer.split()[0] if answer else "yes"

        if verdict.startswith("y"):
            kept.append(repo)
            repo["hardware_reason"] = answer
        else:
            dropped.append(full)

    logger.info(f"[Deps] kept {len(kept)}/{len(candidates)}  (dropped {len(dropped)})")
    state.hardware_filtered = kept
    return {"hardware_filtered": kept}
