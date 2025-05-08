# tools/convert_query.py
import logging
from tools.chat import iterative_convert_to_search_tags
from tools.parse_hardware import parse_hardware_spec

logger = logging.getLogger(__name__)

def convert_searchable_query(state, config):
    # 1) Extract hardware_spec so we can remove it from the tags
    parse_hardware_spec(state, config)
    hw = getattr(state, "hardware_spec", None) or ""

    # 2) Generate the raw colon-separated tags
    user_query = getattr(state, "user_query", None)
    llm_config = getattr(state, "llm_config", None)
    if not user_query or not llm_config:
        raise ValueError("State must have user_query and llm_config attributes")
    raw = iterative_convert_to_search_tags(user_query, llm_config)

    # 3) Filter out any tag that matches the hardware spec token
    filtered = [tag for tag in raw.split(":") if tag and tag != hw]
    searchable = ":".join(filtered)

    # 4) Store and log the cleaned searchable query
    state.searchable_query = searchable
    logger.info(f"Converted searchable query (hardware removed): {searchable}")
    return {"searchable_query": searchable}
