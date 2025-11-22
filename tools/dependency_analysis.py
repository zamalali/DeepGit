# tools/dependency_analysis.py
import os
import asyncio
import logging
import base64
import toml
import httpx
from async_lru import alru_cache
from tools.chat import chain
from tools.mcp_adapter import mcp_adapter
import time
from aiolimiter import AsyncLimiter
logger = logging.getLogger(__name__)
LLM_RATE_LIMITER = AsyncLimiter(1, time_period=30.0)

# ==================================================
# ✅ 异步 + 缓存版本：获取 GitHub 文件内容
# ==================================================
@alru_cache(maxsize=1024)
async def _gh_raw(owner: str, repo: str, path: str, token: str) -> str | None:
    """
    异步获取 GitHub 仓库文件（带缓存），自动解析 base64。
    """
    MAX_RETRIES = 3
    INITIAL_DELAY = 2  # 初始退避延迟
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Authorization": f"token {token}"}
    for attempt in range(MAX_RETRIES):
        try:
            r = await mcp_adapter.fetch(url, headers=headers)
            status_code = r.status_code
            #文件成功或不存在时，立即返回结果
            if status_code == 200:
                # 解析内容并返回
                data = r.json()
                if data.get("encoding") == "base64":
                    # 注意：如果 base64 解码是 CPU 密集型，最好使用 asyncio.to_thread
                    return base64.b64decode(data["content"]).decode("utf-8")
                return data.get("content", "")
            
            if status_code == 404:
                # 文件不存在，立即停止并返回 None
                return None
            
            # ----------------------------------------------------
            # 2. 处理 403 / 429 错误 (速率限制)
            # ----------------------------------------------------
            if r.status_code == 403 or r.status_code == 429:
                reset_time_str = r.headers.get("X-Ratelimit-Reset")
                current_time = int(time.time())
                
                # 计算等待时间 (优先使用 GitHub 提供的重置时间)
                if reset_time_str:
                    reset_time = int(reset_time_str)
                    delay = max(reset_time - current_time, INITIAL_DELAY * (2 ** attempt))
                else:
                    # 如果头部缺失，使用指数退避
                    delay = INITIAL_DELAY * (2 ** attempt)

                logger.warning(
                    f"GitHub Rate Limit Hit. Waiting {delay:.2f}s before retrying {path}. (Attempt {attempt + 1})"
                )
                await asyncio.sleep(delay)
                continue # 进入下一次循环重试
            logger.warning(
                f"GitHub API unexpected error {status_code}. Retrying with delay. (Attempt {attempt + 1})"
            )
            # 使用简单的指数退避等待
            delay = INITIAL_DELAY * (2 ** attempt)
            await asyncio.sleep(delay)
            continue # 进入下一次循环重试
            
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.NetworkError) as e:
            # 捕获网络异常 (也属于瞬时错误)
            logger.error(f"Network error ({type(e).__name__}) fetching {owner}/{repo}/{path}. Retrying.")
            
            # 应用指数退避等待
            delay = INITIAL_DELAY * (2 ** attempt)
            await asyncio.sleep(delay)
            continue
            
    # ----------------------------------------------------
    # 4. 达到最大重试次数
    # ----------------------------------------------------
    logger.error(f"Failed to fetch {owner}/{repo}/{path} after {MAX_RETRIES} attempts.")
    return None

           



# ==================================================
# ✅ 异步：并发抓取 pyproject.toml 和 requirements.txt
# ==================================================
async def _collect_deps(owner: str, repo: str, token: str) -> list[str]:
    """
    并发抓取依赖文件，返回依赖列表。
    """
    reqs_task = _gh_raw(owner, repo, "requirements.txt", token)
    py_task = _gh_raw(owner, repo, "pyproject.toml", token)

    reqs, py = await asyncio.gather(reqs_task, py_task)

    deps = []
    if reqs:
        deps += [l.strip() for l in reqs.splitlines() if l.strip() and not l.startswith("#")]
    if py:
        try:
            deps += list(toml.loads(py).get("tool", {}).get("poetry", {}).get("dependencies", {}).keys())
        except Exception as e:
            logger.warning(f"Failed to parse pyproject.toml in {owner}/{repo}: {e}")
    return deps

async def get_llm_compatibility(prompt: str):
    # 将同步的 chain.invoke 移到线程中执行
    # 注意：如果 chain.invoke/ainvoke 内部已经有重试机制，这里可以直接调用。
    # 如果只有同步的 chain.invoke，我们使用 asyncio.to_thread 来运行它
    
    async with LLM_RATE_LIMITER:#这个语句的意思是，进入这个代码块之前，必须从限流器获得许可
    # 假设你的同步 LLM 调用接口是 tools.chat.chain.invoke
        try:
        # 使用 asyncio.to_thread 运行同步函数
            raw_output = await asyncio.to_thread(chain.invoke, {"query": prompt})
        
        # 解析输出
            ans = raw_output.content.strip()
            ans = ans.split()[0].upper() if ans else "NO"
            return ans
        except Exception as e:
            logger.error(f"LLM failed via thread isolation: {e}")
        # 如果 LLM 失败，默认不兼容
            return "NO"


# ==================================================
# ✅ 主流程：并发执行依赖分析
# ==================================================
async def dependency_analysis_async(state, config):
    """
    异步版本依赖分析。
    从 GitHub 仓库抓取依赖文件，根据硬件兼容性筛选候选项目。
    """
    hw = state.hardware_spec
    cand = state.filtered_candidates

    if not hw:
        state.hardware_filtered = cand
        return {"hardware_filtered": cand}

    token = os.getenv("GITHUB_API_KEY", "")
    kept = []

    # 限制并发数，避免 GitHub API 速率封锁
    sem = asyncio.Semaphore(5)

    async def process_repo(repo):
        full = repo.get("full_name", "")
        if "/" not in full:
            return repo

        o, n = full.split("/", 1)
        async with sem:
            deps = await _collect_deps(o, n, token)

        if not deps:  # 没依赖文件，默认轻量可兼容
            return repo

        # 构造 prompt
        prompt = (
            f"Given the following dependency list, can this project run on {hw}? "
            f"Answer YES or NO and a short reason.\n\nDependencies:\n{', '.join(deps[:25])}"
        )
        ans = await get_llm_compatibility(prompt)


        if ans == "YES":
            return repo
        else:
            logger.info(f"[Deps] drop {full} for {hw}. LLM Ans: {ans}")
            return None

    # 并发执行所有仓库依赖分析
    results = await asyncio.gather(*[process_repo(r) for r in cand])
    kept = [r for r in results if r is not None]

    state.hardware_filtered = kept
    return {"hardware_filtered": kept}
def dependency_analysis(state, config):
    return asyncio.run(dependency_analysis_async(state, config))