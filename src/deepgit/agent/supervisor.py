from __future__ import annotations

import json
import logging

from deepgit.agent.tools import TOOLS, reset_candidates
from deepgit.config import get_settings
from deepgit.lm import configure_dspy
from deepgit.reasoning import IntentExpander, ReportSynthesizer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are DeepGit, an expert at finding the *best-fit* GitHub \
repositories for a developer's need — including hidden gems, not just popular repos.

You have tools to search GitHub, list a repo's files, read the actual source/test \
files, and judge how well a repo fits the intent. Your edge over a plain search is \
that you READ THE CODE before deciding.

Follow this method:
1. Plan: break the intent into hard requirements (must_have), soft preferences, and \
anti-patterns (things to avoid). Consider which ecosystems/languages fit.
2. Search: call github_search several times with different, targeted queries. Use \
GitHub qualifiers (language:, stars:>N, topic:) to widen and narrow coverage.
3. Shortlist: from the candidates, pick the 4-5 most promising. Do not over-rely on \
stars — a smaller repo can be the best fit. Do NOT shortlist more than 5.
4. Verify: for each shortlisted repo, make ONE read_repo_files call listing the key \
files you need (the manifest such as pyproject.toml/package.json, the core module, \
and one test file). Read them in a single batched call per repo — do not read files \
one at a time. This is how you confirm it truly does what the user needs.
5. Judge: call judge_repo_fit for each verified repo, passing the same intent, \
must_have, and anti_patterns.
6. Recommend: rank by genuine fit. Clearly state the top pick and when an \
alternative is better, citing concrete evidence from the code you read.

Be efficient with tool calls, but never skip reading code for your top candidates.
When done, produce a concise Markdown report ranking the best repositories with \
reasons and any risks (maintenance, license, maturity)."""


def _build_model():
    """Construct the chat model for the supervisor from settings."""
    settings = get_settings()
    if settings.llm_provider == "vertex_ai":
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model=settings.llm_model,
            project=settings.vertex_project or None,
            location=settings.vertex_location or "us-central1",
            temperature=settings.llm_temperature,
        )
    # Fallback: let deepagents resolve a provider:model string.
    provider = "openai" if settings.llm_provider == "openai" else settings.llm_provider
    return f"{provider}:{settings.llm_model}"


def build_supervisor():
    """Build (and compile) the DeepGit supervisor agent."""
    configure_dspy()
    from deepagents import create_deep_agent

    return create_deep_agent(
        _build_model(),
        TOOLS,
        system_prompt=SYSTEM_PROMPT,
        name="deepgit-supervisor",
    )


def _plan_hint(intent: str) -> str:
    """Pre-compute a search plan to seed the agent (best-effort)."""
    try:
        plan = IntentExpander()(intent=intent)
        return json.dumps(plan.model_dump(), ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 - planning is advisory
        logger.warning("IntentExpander failed, agent will plan on its own: %s", exc)
        return ""


def search(intent: str, *, recursion_limit: int = 120, stream: bool = True) -> str:
    """Run a full DeepGit search for ``intent`` and return a Markdown report.

    When ``stream`` is True (default) the agent's steps are streamed and logged
    as they happen, so progress is visible instead of a long silent wait.
    """
    reset_candidates()
    logger.info("[deepgit] building supervisor ...")
    agent = build_supervisor()

    plan_hint = _plan_hint(intent)
    if plan_hint:
        logger.info("[deepgit] seed plan: %s", plan_hint[:400])
    user_msg = f"Find the best GitHub repositories for this need:\n\n{intent}"
    if plan_hint:
        user_msg += f"\n\nSuggested search plan (refine as needed):\n{plan_hint}"

    payload = {"messages": [{"role": "user", "content": user_msg}]}
    config = {"recursion_limit": recursion_limit}

    if not stream:
        result = agent.invoke(payload, config=config)
        return _final_text(result)

    logger.info("[deepgit] running agent (streaming) ...")
    last_state: dict = {}
    step = 0
    for chunk in agent.stream(payload, config=config, stream_mode="values"):
        last_state = chunk
        msgs = chunk.get("messages", [])
        if not msgs:
            continue
        msg = msgs[-1]
        step += 1
        _log_step(step, msg)
    logger.info("[deepgit] agent finished after %s streamed states", step)
    return _final_text(last_state)


def _log_step(step: int, msg) -> None:
    """Log a single streamed message (tool call, tool result, or assistant text)."""
    role = getattr(msg, "type", getattr(msg, "role", "?"))
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            preview = json.dumps(args, ensure_ascii=False)[:160]
            logger.info("[step %s] tool-call: %s(%s)", step, name, preview)
        return
    if role in ("tool", "ToolMessage"):
        content = _content_to_text(getattr(msg, "content", ""))
        logger.info("[step %s] tool-result: %s", step, content[:200].replace("\n", " "))
        return
    text = _content_to_text(getattr(msg, "content", "") or "")
    if text.strip():
        logger.info("[step %s] assistant: %s", step, text[:200].replace("\n", " "))


def _final_text(state: dict) -> str:
    messages = state.get("messages", []) if isinstance(state, dict) else []
    if not messages:
        return "(no result)"
    final = messages[-1]
    return _content_to_text(getattr(final, "content", None) or final)


def _content_to_text(content) -> str:
    """Normalise LangChain message content (str | list of parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
        return "\n".join(p for p in parts if p)
    return str(content)


def search_with_synthesis(intent: str, **kwargs) -> dict:
    """Run the agent, then apply the DSPy report synthesizer over its verdicts."""
    from deepgit.agent.tools import all_candidates

    report = search(intent, **kwargs)
    judged = [
        {
            "repo": r.name_with_owner,
            "stars": r.stars,
            "language": r.primary_language,
            "description": r.description[:160],
        }
        for r in all_candidates()
    ]
    synthesized = ""
    top_pick = ""
    try:
        pred = ReportSynthesizer()(
            intent=intent, judged_repos=json.dumps(judged, ensure_ascii=False)
        )
        synthesized = pred.report
        top_pick = pred.top_pick
    except Exception as exc:  # noqa: BLE001
        logger.warning("ReportSynthesizer failed: %s", exc)
    return {
        "agent_report": report,
        "synthesized_report": synthesized,
        "top_pick": top_pick,
        "candidate_count": len(judged),
    }
