
"""
根据用户需求，生成研究简报。

该模块会创建”研究范围界定“的工作流，如下：
1. 判断用户需求是否清晰；
2. 根据对话，生成研究简报。
"""
import os
from dotenv import load_dotenv

from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research_from_scratch.prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt
)
from deep_research_from_scratch.state_scope import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ResearchQuestion
)

# ==== 工具函数 ====

def get_today_str() -> str:
    """获取当前日期字符串"""
    # 使用#代替-,避免跨平台问题
    return datetime.now().strftime("%a %b %#d, %Y")

# ==== 配置 ====

# 加载环境变量
load_dotenv()

# 初始化gpt模型
api_url = os.getenv('KIMI_API_URL')
api_key = os.getenv('KIMI_API_KEY')
model_name = os.getenv('KIMI_MODEL')
model = init_chat_model(
    model_provider="openai",  # 避免langchain根据模型名自动选择供应商
    model=model_name, 
    temperature=0.0,
    api_key=api_key,
    base_url=api_url
)

# ==== 工作流的结点 ====

def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", END]]:
    """
    判断是否用户的请求信息包含足够的信息，用于生成研究简报。

    使用结构化的输出进行判断，同时避免幻觉。
    路由到研究简报生成，或生成一个需要用户澄清的问题。
    """
    # 结构化输出模型(相当于打开了模型的json_format开关，并转换为pydantic的schema)
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # 使用澄清用指令调用模型
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]),
            date=get_today_str()
        ))
    ])

    # 根据need_clarification，判断后续的流程，并更新state
    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_research_brief(state: AgentState):
    """
    将会话内容转换为研究简报。

    使用结构化的输出，保证简报包含所有生成研究报告需要的细节和信息。
    """
    # 设置结构化输出模型
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # 生成研究简报
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # 更新state，生成研究简报并传递给管理者agent(supervisor)
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }

# ==== 构建研究范围界定的工作流（基于langgraph） ====

# 创建工作流
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# 添加工作流的结点（nodes）
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

# 添加工作流的边（edges）
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

# 编译工作流
scope_research = deep_researcher_builder.compile()
