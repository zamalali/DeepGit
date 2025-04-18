# -*- coding: utf-8 -*-
import logging
import re
import string
from tools.chat import chain       # Re‑use your ChatGroq pipeline

logger = logging.getLogger(__name__)

# Canonical tokens and a small alias set for each (lower‑case, no punctuation)
ALIAS_TABLE = {
    "cpu-only": {
        "cpuonly", "cpu only", "cpu‑only", "cpu", "no gpu", "nogpu",
        "gpu poor", "gpu‑poor", "lightweight"
    },
    "low-memory": {
        "lowmemory", "low memory", "small memory", "tiny ram", "low‑ram",
        "≤4gb ram", "under 4gb"
    },
    "mobile": {
        "mobile", "edge", "phone", "android", "raspberry", "raspberrypi",
        "raspberry pi"
    },
}

PROMPT = (
    "A user is describing a software project they want to run.\n"
    "Extract the *single best* hardware class they implicitly or explicitly require.\n"
    "Reply with exactly one token from this list (no extra text):\n"
    "cpu-only   – commodity CPUs (no dedicated GPU)\n"
    "low-memory – ≤ 4 GB RAM devices\n"
    "mobile     – phones / edge devices / Raspberry Pi\n"
    "none       – no obvious hardware restriction\n\n"
    "User query: ```{query}```"
)

# --------------------------------------------------------------------------- #
# Helper – normalise a phrase to “alias” form (lower, delete punctuation/WS)
# --------------------------------------------------------------------------- #
_normaliser = str.maketrans("", "", string.punctuation)
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_normaliser).lower()).strip()

# --------------------------------------------------------------------------- #
# Main entry‑point used by the LangGraph node
# --------------------------------------------------------------------------- #
def parse_hardware_spec(state, config):
    """Populate state.hardware_spec (None if no constraint)."""
    user_q = state.user_query

    # ---------- 1) quick heuristic scan ----------------------------------- #
    norm_q = _norm(user_q)
    for canonical, aliases in ALIAS_TABLE.items():
        if any(alias in norm_q for alias in aliases):
            logger.info(f"[Hardware] heuristic → {canonical}")
            state.hardware_spec = canonical
            return {"hardware_spec": canonical}

    # ---------- 2) ML fallback ------------------------------------------- #
    model_resp = chain.invoke({"query": PROMPT.format(query=user_q)})
    resp_token = _norm(model_resp.content)

    canonical = next(
        (key for key, aliases in ALIAS_TABLE.items() if resp_token in aliases or resp_token == key),
        None,
    )

    state.hardware_spec = canonical
    logger.info(f"[Hardware] model → {canonical}")
    return {"hardware_spec": canonical}
