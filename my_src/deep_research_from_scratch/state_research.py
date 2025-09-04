
"""定义用于Research Agent的state和pydantic的schemas"""

import operator
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ==== 定义state ====

class ResearcherState(TypedDict):
    """用于存储上下文信息"""
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]

class ResearcherOutputState(TypedDict):
    """用于存储Research Agent的输出"""
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

# ==== 定义schemas ====

class ClarifyWithUser(BaseModel):
    """scoping澄清阶段的schema定义"""
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question."
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information."
    )

class ResearchQuestion(BaseModel):
    """用于生成研究简报的schema定义"""
    research_brief: str = Field(
        description="A research question that will be used to guide the research."
    )

class Summary(BaseModel):
    """用于网页内容总结的schema定义"""
    summary: str = Field(
        description="Concise summary of the webpage content"
    )
    key_excerpts: str = Field(
        description="Important quotes and excerpts from the content"
    )
