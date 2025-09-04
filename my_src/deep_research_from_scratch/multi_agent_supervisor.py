
"""协调多agent的研究监督者(Multi-Agent Supervisor)"""

import os
import asyncio

from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research_from_scratch.prompts import lead_researcher_prompt
from deep_research_from_scratch.research_agent import researcher_agent
from deep_research_from_scratch.state_multi_agent_supervisor import (
    SupervisorState,
    ConductResearch,
    ResearchComplete
)
from deep_research_from_scratch.utils import get_today_str, think_tool

def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """从supervisor的messages中提取出tool_calls中的研究要点信息(包括总结后的研究报告)"""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

# 用于notebook中的异步执行
try:
    import nest_asyncio
    # 仅在notebook中使用
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # 在其他环境中运行时，忽略此错误
except ImportError:
    pass

# ==== 配置 ====

supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
# 模型初始化
api_url = os.getenv('KIMI_API_URL')
api_key = os.getenv('KIMI_API_KEY')
model_name = os.getenv('KIMI_MODEL')
supervisor_model = init_chat_model(
    model_provider="openai",  # 避免langchain根据模型名自动选择供应商
    model=model_name, 
    # temperature=0.0,
    api_key=api_key,
    base_url=api_url,
    # max_tokens=64000
)
# 绑定工具
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

# 最大迭代次数(工具调用次数)
max_researcher_iterations = 6

# 最大并行研究数
max_concurrent_researchers = 3

# ==== supervisor节点 ====

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """协调research过程"""
    supervisor_messages = state.get("supervisor_messages", [])

    # system message
    system_message = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    # 决定下一步做什么
    response = await supervisor_model_with_tools.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", END]]:
    """supervisor工具节点,执行工具调用"""
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    tool_messages = []
    all_raw_notes = []
    next_step = "supervisor"  # 默认下一步是supervisor
    should_end = False

    # 添加调试信息
    print(f"🔍 当前迭代次数: {research_iterations}, 最大限制: {max_researcher_iterations}")
    print(f"🔍 最新消息是否有tool_calls: {bool(most_recent_message.tool_calls)}")

    # 优先判断是否需要结束
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    print(f"🔍 超过迭代限制: {exceeded_iterations}")
    print(f"🔍 没有tool_calls: {no_tool_calls}")
    print(f"🔍 研究完成: {research_complete}")

    if exceeded_iterations or no_tool_calls or research_complete:
        should_end = True
        next_step = END
        print(f"✅ 满足结束条件，准备结束流程")
    else:
        print(f"➡️ 继续执行，下一步: {next_step}")
        print(f"🔍 当前消息: {most_recent_message}")
        # 工具执行
        try:
            # 分离tool调用
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            # 执行think_tool调用(同步执行)
            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # 执行conduct_research调用(异步执行)
            if conduct_research_calls:
                # 并发执行sub-agent
                coros = [
                    researcher_agent.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"]
                    })
                    for tool_call in conduct_research_calls
                ]

                # 等待所有异步执行完成
                tool_results = await asyncio.gather(*coros)

                # 提取总结的研究报告
                research_tool_messages = [
                    ToolMessage(
                        content=result.get("compressed_research", "Error synthesizing research report"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for result, tool_call in zip(tool_results, conduct_research_calls)
                ]

                tool_messages.extend(research_tool_messages)

                # 合并raw_notes
                all_raw_notes = [
                    "\n".join(result.get("raw_notes", []))
                    for result in tool_results
                ]
        except Exception as e:
            print(f"Error in supervisor tools: {e}")
            should_end = True
            next_step = END

    if should_end:
        return Command(
            goto=next_step,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    else:
        return Command(
            goto=next_step,
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": all_raw_notes
            }
        )

# ==== 构建工作流 ====

# 构建supervisor的graph
# 添加节点
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)

# 添加边
supervisor_builder.add_edge(START, "supervisor")

# 编译
supervisor_agent = supervisor_builder.compile()
