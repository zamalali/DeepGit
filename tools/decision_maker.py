# tools/decision_maker.py
import logging
from tools.decision import should_run_code_analysis

logger = logging.getLogger(__name__)

def decision_maker(state, config):
    repo_count = len(state.filtered_candidates)
    decision = should_run_code_analysis(state.user_query, repo_count)
    state.run_code_analysis = (decision == 1)
    logger.info(f"Decision Maker: run_code_analysis = {state.run_code_analysis}")
    return {"run_code_analysis": state.run_code_analysis}
