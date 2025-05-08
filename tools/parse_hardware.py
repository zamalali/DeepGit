# tools/parse_hardware_spec.py
import re, logging
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

VALID_SPECS = ("cpu-only", "low-memory", "mobile")

HARDWARE_PATTERNS = {
    "cpu-only":   [r"cpu[- ]only", r"no[- ]?gpu",  r"gpu[- ]poor", r"lightweight"],
    "low-memory": [r"low[- ]?memory", r"small[- ]?memory"],
    "mobile":     [r"mobile", r"raspberry", r"android"],
}

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a hardware constraint detection expert."),
    ("human", "Extract any hardware constraints from the user query. Return exactly one of: cpu-only, low-memory, mobile, NONE.\n\nUser query:\n{query}")
])

def parse_hardware_spec(state, config):
    user_query = getattr(state, "user_query", None)
    llm_config = getattr(state, "llm_config", None)
    if not user_query or not llm_config:
        raise ValueError("State must have user_query and llm_config attributes")
    
    q = user_query.lower()

    # 1) Fast heuristic
    for spec, patterns in HARDWARE_PATTERNS.items():
        if any(re.search(pat, q) for pat in patterns):
            logger.info(f"[Hardware] regex -> {spec}")
            state.hardware_spec = spec
            return {"hardware_spec": spec}

    # 2) LLM fallback
    llm = llm_config.get_llm(temperature=0.3, max_tokens=128)
    chain = prompt | llm
    resp = chain.invoke({"query": user_query}).content.strip().lower()
    spec = resp if resp in VALID_SPECS else None
    logger.info(f"[Hardware] LLM  -> {spec}")
    state.hardware_spec = spec
    return {"hardware_spec": spec}
