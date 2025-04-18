# tools/parse_hardware_spec.py
import re, logging
from tools.chat import chain 

logger = logging.getLogger(__name__)

VALID_SPECS = ("cpu-only", "low-memory", "mobile")

HARDWARE_PATTERNS = {
    "cpu-only":   [r"cpu[- ]only", r"no[- ]?gpu",  r"gpu[- ]poor", r"lightweight"],
    "low-memory": [r"low[- ]?memory", r"small[- ]?memory"],
    "mobile":     [r"mobile", r"raspberry", r"android"],
}

PROMPT_TEMPLATE = (
    "Extract any hardware constraints from the user query. "
    "Return exactly one of: cpu-only, low-memory, mobile, NONE."
)

def parse_hardware_spec(state, config):
    q = state.user_query.lower()

    # 1) Fast heuristic
    for spec, patterns in HARDWARE_PATTERNS.items():
        if any(re.search(pat, q) for pat in patterns):
            logger.info(f"[Hardware] regex -> {spec}")
            state.hardware_spec = spec
            return {"hardware_spec": spec}

    # 2) LLM fallback
    full = f"{PROMPT_TEMPLATE}\n\nUser query:\n{state.user_query}"
    resp = chain.invoke({"query": full}).content.strip().lower()
    spec = resp if resp in VALID_SPECS else None
    logger.info(f"[Hardware] LLM  -> {spec}")
    state.hardware_spec = spec
    return {"hardware_spec": spec}
