# tools/convert_query.py
import logging
from tools.chat import iterative_convert_to_search_tags
from tools.parse_hardware import parse_hardware_spec

logger = logging.getLogger(__name__)

def convert_searchable_query(state, config):
    # 1) Extract hardware_spec so we can remove it from the tags
    parse_hardware_spec(state, config)
    hw = state.hardware_spec or ""

    # 2) Generate the raw colon-separated tags
    raw = iterative_convert_to_search_tags(state.user_query)

    # 3) Filter out any tag that matches the hardware spec token
    filtered = [tag for tag in raw.split(":") if tag and tag != hw]
    searchable = ":".join(filtered)

    # 4) Store and log the cleaned searchable query
    state.searchable_query = searchable
    logger.info(f"Converted searchable query (hardware removed): {searchable}")
    return {"searchable_query": searchable}
