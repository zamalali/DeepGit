# tools/convert_query.py
import logging
from tools.chat import iterative_convert_to_search_tags

logger = logging.getLogger(__name__)

def convert_searchable_query(state, config):
    searchable = iterative_convert_to_search_tags(state.user_query)
    state.searchable_query = searchable
    logger.info(f"Converted searchable query: {searchable}")
    return {"searchable_query": searchable}
