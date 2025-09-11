
"""Multi-Agent Research System"""

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from deep_research_from_scratch.utils import get_today_str
from deep_research_from_scratch.my_prompts import final_report_generation_prompt
from deep_research_from_scratch.state_scope import AgentState, AgentInputState
from deep_research_from_scratch.research_agent_scope import clarify_with_user, write_research_brief
from deep_research_from_scratch.multi_agent_supervisor import supervisor_agent

# ==== Config ====

import os
from langchain.chat_models import init_chat_model
# 模型初始化
api_url = os.getenv('ARK_API_URL')
api_key = os.getenv('ARK_API_KEY')
model_name = os.getenv('ARK_MODEL')
writer_model = init_chat_model(
    model_provider="openai",  # 避免langchain根据模型名自动选择供应商
    model=model_name, 
    # temperature=0.0,
    api_key=api_key,
    base_url=api_url,
    max_tokens=32000
)

# ==== 最终报告生成 ====

from deep_research_from_scratch.state_scope import AgentState

async def final_report_generation(state: AgentState):
    """最终报告生成"""

    notes = state.get("notes", [])

    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        format_constraints=state.get("format_constraints", ""),
        date=get_today_str()
    )

    final_report = await writer_model.ainvoke(
        [HumanMessage(content=final_report_prompt)]
    )

    return {
        "final_report": final_report.content,
        "messages": ["Here is the final report: \n" + final_report.content]
    }

# ==== 构建graph ====

# 构建工作流
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# 添加节点
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# 添加edge
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# 编译
agent = deep_researcher_builder.compile()
