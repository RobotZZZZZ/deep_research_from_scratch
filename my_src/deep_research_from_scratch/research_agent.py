
"""实现Research Agent，通过多次搜索和整合回答研究问题"""

import os
from pydantic import BaseModel, Field
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain.chat_models import init_chat_model

from deep_research_from_scratch.state_research import ResearcherState, ResearcherOutputState
from deep_research_from_scratch.utils import tavily_search, get_today_str, think_tool
from deep_research_from_scratch.prompts import research_agent_prompt, compress_research_system_prompt, compress_research_human_message

# ==== 配置 ====

# 工具设置
tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

# 初始化模型
api_url = os.getenv('KIMI_API_URL')
api_key = os.getenv('KIMI_API_KEY')
model_name = os.getenv('KIMI_MODEL')
model = init_chat_model(
    model_provider="openai",  # 避免langchain根据模型名自动选择供应商
    model=model_name, 
    # temperature=0.0,
    api_key=api_key,
    base_url=api_url
)
# 工具绑定
model_with_tools = model.bind_tools(tools)
summarization_model = init_chat_model(
    model_provider="openai",  # 避免langchain根据模型名自动选择供应商
    model=model_name, 
    # temperature=0.0,
    api_key=api_key,
    base_url=api_url
)
compress_model = init_chat_model(
    model_provider="openai",  # 避免langchain根据模型名自动选择供应商
    model=model_name, 
    # temperature=0.0,
    api_key=api_key,
    base_url=api_url,
    max_tokens=64000
)

# ==== agent节点 ====

def llm_call(state: ResearcherState):
    """后续action的判断

    根据当前的对话内容，判断后续的action：
    1. 如果需要继续搜索，则调用tavily_search工具
    2. 根据收集到的信息，给出最终回答
    """
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]
            )
        ]
    }

def tool_node(state: ResearcherState):
    """工具执行"""
    tool_calls = state["researcher_messages"][-1].tool_calls

    # 执行工具
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    # 工具调用的输出结果
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}

def compress_research(state: ResearcherState) -> dict:
    """压缩research相关的信息"""

    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] \
        + state.get("researcher_messages", []) \
        + [HumanMessage(content=compress_research_human_message)]
    response = compress_model.invoke(messages)

    # 保留原始工具调用信息和LLM的输出
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
    """根据当前的对话内容，判断是否需要继续research的循环"""
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # 如果最后一条消息是tool_message，则继续research
    if last_message.tool_calls:
        return "tool_node"
    return "compress_research"

# ==== 定义graph ====

# 创建agent的工作流
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# 添加节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# 添加边（edges）
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,  # 判断条件
    {
        "tool_node": "tool_node",
        "compress_research": "compress_research",
    }
)
agent_builder.add_edge("tool_node", "llm_call")
agent_builder.add_edge("compress_research", END)

# 编译
researcher_agent = agent_builder.compile()
