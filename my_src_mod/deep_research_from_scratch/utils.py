
"""Research工具定义"""

import os
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing_extensions import Annotated, List, Literal

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from tavily import AsyncTavilyClient

from deep_research_from_scratch.state_research import Summary
from deep_research_from_scratch.my_prompts import summarize_webpage_prompt

# ==== 常用函数定义 ====

def get_today_str() -> str:
    """获取当前日期字符串"""
    # 使用#代替-,避免跨平台问题
    return datetime.now().strftime("%a %b %#d, %Y")

def get_current_dir() -> Path:
    """获取当前模块的所在目录"""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

# ==== 配置 ====

# 本地MCP服务端-用于访问本地文件
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",  # 需要的话，自动安装
            "@modelcontextprotocol/server-filesystem",
            str(get_current_dir() / "files")  # 使用当前目录下的files文件夹
        ],
        "transport": "stdio",  # 使用stdio传输
        "env": {
            "LANG": "zh_CN.UTF-8",
            "LC_ALL": "zh_CN.UTF-8",
            "LC_CTYPE": "zh_CN.UTF-8"
        }
    }
}

# 全局的client，仅在需要的时候lazy初始化
_client = None

def get_mcp_client():
    """Lazy initialize the MCP client"""
    global _client
    if _client is None:
        _client = MultiServerMCPClient(mcp_config)
    return _client

# 初始化gpt模型
api_url = os.getenv('ARK_API_URL')
api_key = os.getenv('ARK_API_KEY')
model_name = os.getenv('ARK_MODEL')
summarization_model = init_chat_model(
    model_provider="openai",  # 避免langchain根据模型名自动选择供应商
    model=model_name, 
    # temperature=0.0,
    api_key=api_key,
    base_url=api_url,
    max_tokens=32000
)
# 初始化tavily客户端
tavily_client = AsyncTavilyClient()

# ==== 搜索功能相关函数定义 ====

async def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> List[dict]:
    """
    使用tavily搜索多个查询

    Args:
        search_queries: 搜索查询列表
        max_results: 每个查询的最大结果数
        topic: 搜索主题
        include_raw_content: 是否包含原始网页内容
    Returns:
        List[dict]: 搜索结果列表, 每个结果包含url, title, snippet, raw_content
    """
    # 依次进行搜索（也可以使用AsyncTavilyClient进行并发搜索）
    search_docs = []
    coros = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        ) for query in search_queries
    ]
    search_docs = await asyncio.gather(*coros)

    return search_docs

def deduplicate_search_results(search_results: List[dict]) -> dict:
    """根据url去重"""
    unique_results = {}

    for response in search_results:
        for result in response["results"]:
            url = result["url"]
            if url not in unique_results:
                unique_results[url] = result

    return unique_results

async def summarize_webpage_content(webpage_content: str) -> str:
    """
    使用LLM对网页内容进行总结和要点摘录

    Args:
        webpage_content: 网页内容
    Returns:
        str: 总结
    """
    try:
        # 检查内容长度，如果超过API限制则截断
        max_length = 80000  # 留一些缓冲空间，低于98304的限制
        if len(webpage_content) > max_length:
            # 尝试找到一个自然的截断点（段落结尾）
            truncated_content = webpage_content[:max_length]
            last_period = truncated_content.rfind('.')
            last_newline = truncated_content.rfind('\n')
            # 选择最近的段落或行结束点
            cutoff = max(last_period, last_newline) if max(last_period, last_newline) > max_length * 0.8 else max_length
            webpage_content = webpage_content[:cutoff] + "..."
            print(f"网页内容过长，已截断至 {len(webpage_content)} 字符")

        structured_model = summarization_model.with_structured_output(Summary)
        summary = await structured_model.ainvoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=webpage_content,
                date=get_today_str()
            ))
        ])

        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        return formatted_summary

    except Exception as e:
        print(f"总结网页时发生错误: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content

async def process_search_results(unique_results: dict) -> dict:
    """
    处理搜索结果

    Args:
        unique_results: 去重后的搜索结果
    Returns:
        dict: 处理后的搜索结果
    """
    async def process_search_result_single(url: str, result: str):
        if not result.get("raw_content"):
            content = result["content"]
        else:
            # 总结内容
            content = await summarize_webpage_content(result["raw_content"])
        return url, {"title": result["title"], "content": content,}

    coros = [
        process_search_result_single(url, result)
        for url, result in unique_results.items()
    ]
    summarized_results_list = await asyncio.gather(*coros)

    return dict(summarized_results_list)

def format_search_output(summarized_results: dict) -> str:
    """
    格式化搜索结果

    Args:
        summarized_results: 处理后的搜索结果
    Returns:
        str: 格式化后的搜索结果
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i} {result["title"]} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result["content"]}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output

# ==== 定义tools ====

# 搜索工具
@tool(parse_docstring=True)
async def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """
    # 单次搜索
    print(f"当前搜索query: {query}")
    search_results = await tavily_search_multiple(
        [query],
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # 结果去重
    unique_results = deduplicate_search_results(search_results)

    # 总结页面内容
    summarized_results = await process_search_results(unique_results)

    # 格式化搜索结果
    formatted_output = format_search_output(summarized_results)

    return formatted_output

# 反思工具
# 有意思的是，反思工具的功能是通过docstring的描述定义的（效果类似于prompt）
@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """

    return f"Reflection recorded: {reflection}"
