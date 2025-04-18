# tools/dependency_analysis.py
import os, logging, httpx, toml, base64
from functools import lru_cache
from tools.chat import chain     

logger = logging.getLogger(__name__)

QUESTION_TMPL = (
    "Given the following dependency list, can this project run on {hw}? "
    "Answer YES or NO and a short reason.\n\nDependencies:\n{deps}"
)

@lru_cache(maxsize=1024)
def _gh_raw(owner: str, repo: str, path: str, token: str) -> str | None:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    r = httpx.get(url, headers={"Authorization": f"token {token}"})
    if r.status_code != 200:
        return None
    data = r.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]).decode("utf-8")
    return data.get("content", "")

def _collect_deps(owner: str, repo: str, token: str) -> list[str]:
    reqs = _gh_raw(owner, repo, "requirements.txt", token) or ""
    py   = _gh_raw(owner, repo, "pyproject.toml",  token) or ""
    deps = [l.strip() for l in reqs.splitlines() if l.strip() and not l.startswith("#")]
    if py:
        try:
            deps += list(toml.loads(py).get("tool",{}).get("poetry",{}).get("dependencies",{}).keys())
        except Exception:
            pass
    return deps

def dependency_analysis(state, config):
    hw   = state.hardware_spec          # None means “no constraint”
    cand = state.filtered_candidates
    if not hw:
        state.hardware_filtered = cand
        return {"hardware_filtered": cand}

    token = os.getenv("GITHUB_API_KEY", "")
    kept  = []

    for repo in cand:
        full = repo.get("full_name", "")
        if "/" not in full:
            kept.append(repo); continue
        o, n = full.split("/", 1)

        deps = _collect_deps(o, n, token)
        if not deps:                    # empty list = assume lightweight
            kept.append(repo); continue

        prompt = QUESTION_TMPL.format(hw=hw, deps=", ".join(deps[:25]))
        ans    = chain.invoke({"query": prompt}).content.strip().split()[0].upper()
        if ans == "YES":
            kept.append(repo)
        else:
            logger.info(f"[Deps] drop {full} for {hw}")

    state.hardware_filtered = kept
    return {"hardware_filtered": kept}
