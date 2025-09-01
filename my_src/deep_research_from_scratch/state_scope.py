
# -*- coding: utf-8 -*-
"""定义state和Pydantic的schema，用于后续的上下文传递。"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ==== 定义state ====

class AgentInputState(MessagesState):
    """输入state，用于存储用户输入的指令和信息。"""
    pass

class AgentState(MessagesState):
    """用于多agent的上下文传递。"""

    # 研究简报，来自用户对话
    research_brief: Optional[str]
    # multi-agents的场景下，用于管理者(supervisor)和agent之间协调的信息传递
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # 原始的研究报告要点，来自研究阶段
    raw_notes: Annotated[list[str], operator.add] = []
    # 处理后的研究报告要点
    notes: Annotated[list[str], operator.add] = []
    # 最终的研究报告输出
    final_report: str

# ==== 结构后的输出schemas ====

class ClarifyWithUser(BaseModel):
    """schema用于确认用户输入的指令是否清晰。"""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """schema用于生成研究简报"""

    research_brief: str = Field(
        description="A research question that will be used to guide the research."
    )
