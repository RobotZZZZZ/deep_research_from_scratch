
"""Multi-Agent Supervisor State"""

import operator
from typing_extensions import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class SupervisorState(TypedDict):
    """Supervisor State"""

    # 用于在supervisor和researcher之间传递消息
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # 研究简报
    research_brief: str
    # 用于最终报告生成的要点信息
    notes: Annotated[list[str], operator.add] = []
    # 记录迭代次数
    research_iterations: int = 0
    # 原始的来自sub-agent的研究要点
    raw_notes: Annotated[list[str], operator.add] = []

# 这里定义的tool只是用于定义工具的类型（状态记录），而不是用于执行工具
@tool
class ConductResearch(BaseModel):
    """分发工具，用于分发研究任务给researcher"""
    research_topic: str = Field(
        description="The Topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

@tool
class ResearchComplete(BaseModel):
    """研究完成工具，用于指示研究过程完成

    当所有研究任务都完成后调用此工具。
    不需要任何参数，调用本身就表示研究结束。
    """
    pass
