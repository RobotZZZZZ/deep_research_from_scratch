
"""Research Agent with MCP"""

import os

from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END

from deep_research_from_scratch.prompts import (
    research_agent_prompt_with_mcp,
    compress_research_system_prompt, 
    compress_research_human_message
)
from deep_research_from_scratch.state_research import ResearcherState, ResearcherOutputState
from deep_research_from_scratch.utils import get_today_str, think_tool, get_current_dir

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
        "transport": "stdio"  # 使用stdio传输
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

# 模型初始化
api_url = os.getenv('KIMI_API_URL')
api_key = os.getenv('KIMI_API_KEY')
model_name = os.getenv('KIMI_MODEL')
compress_model = init_chat_model(
    model_provider="openai",  # 避免langchain根据模型名自动选择供应商
    model=model_name, 
    # temperature=0.0,
    api_key=api_key,
    base_url=api_url,
    max_tokens=64000
)
model = init_chat_model(
    model_provider="openai",  # 避免langchain根据模型名自动选择供应商
    model=model_name, 
    # temperature=0.0,
    api_key=api_key,
    base_url=api_url,
    max_tokens=64000
)

# ==== agent节点 ====

async def llm_call(state: ResearcherState):
    """根据当前的上下文，判断是否要使用mcp工具"""
    # 获取可用工具
    client = get_mcp_client()
    mcp_tools = await client.get_tools()

    # 添加think_tool
    tools = mcp_tools + [think_tool]

    # 工具绑定
    model_with_tools = model.bind_tools(tools)

    # llm with tools(mcp)
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt_with_mcp.format(date=get_today_str()))]
                + state["researcher_messages"]
            )
        ]
    }

async def tool_node(state: ResearcherState):
    """工具（mcp）执行"""
    # 获取当前llm输出的tool_calls
    tool_calls = state["researcher_messages"][-1].tool_calls

    async def execute_tools():
        """执行工具"""
        # 获取最新的工具信息
        client = get_mcp_client()
        mcp_tools = await client.get_tools()
        tools = mcp_tools + [think_tool]
        tools_by_name = {tool.name: tool for tool in tools}

        # 执行工具
        observations = []
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            if tool_call["name"] == "think_tool":
                # think_tool采用同步执行
                observation = tool.invoke(tool_call["args"])
            else:
                # mcp工具采用异步执行, 使用ainvoke
                observation = await tool.ainvoke(tool_call["args"])
            observations.append(observation)

        # 格式化输出
        tool_outputs = [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            for observation, tool_call in zip(observations, tool_calls)
        ]

        return tool_outputs

    messages = await execute_tools()

    return {"researcher_messages": messages}

def compress_research(state: ResearcherState):
    """总结（压缩）研究报告"""
    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] \
        + state.get("researcher_messages", []) \
        + [HumanMessage(content=compress_research_human_message)]

    response = compress_model.invoke(messages)

    # 保留原始的工具调用和llm输出
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

# ==== 路由逻辑 ====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """根据当前的上下文，判断是否需要继续使用工具或压缩报告"""
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # 如果当前的llm输出中没有tool_calls，则停止使用工具
    if last_message.tool_calls:
        return "tool_node"
    # 否则，进入压缩报告阶段
    return "compress_research"

# ==== 构建工作流（图） ====

# 创建工作流
agent_builder_mcp = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# 添加节点
agent_builder_mcp.add_node("llm_call", llm_call)
agent_builder_mcp.add_node("tool_node", tool_node)
agent_builder_mcp.add_node("compress_research", compress_research)

# 添加边
agent_builder_mcp.add_edge(START, "llm_call")
agent_builder_mcp.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        "compress_research": "compress_research",
    }
)
agent_builder_mcp.add_edge("tool_node", "llm_call")
agent_builder_mcp.add_edge("compress_research", END)

# 编译
agent_mcp = agent_builder_mcp.compile()

