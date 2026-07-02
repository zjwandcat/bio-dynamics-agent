# BioDynamics Agent - SBML 模型解析器
# 使用 LLM 从已有 SBML 模型文本中提取可复用的网络拓扑。

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import llm
from app.prompts import SBML_PARSER_PROMPT


class _SBMLNodeItem(BaseModel):
    id: str = Field(..., description="节点唯一标识符")
    name: str = Field(..., description="节点可读名称")
    type: str = Field(default="Protein", description="节点类型")


class _SBMLEdgeItem(BaseModel):
    source: str = Field(..., description="源节点 id")
    target: str = Field(..., description="目标节点 id")
    interaction: str = Field(default="activation", description="相互作用类型")


class SBMLParseOutput(BaseModel):
    """SBML 模型解析的结构化输出。"""

    is_reusable: bool = Field(default=False, description="模型是否与用户需求高度匹配")
    reuse_reason: str = Field(default="", description="复用理由或放弃理由")
    nodes: list[_SBMLNodeItem] = Field(default_factory=list, description="模型包含的节点")
    edges: list[_SBMLEdgeItem] = Field(default_factory=list, description="模型包含的相互作用")


def parse_sbml_model(
    user_query: str,
    sbml_text: str,
    callbacks: list | None = None,
) -> dict:
    """解析 SBML 模型文本，返回可复用的网络结构字典。

    Args:
        user_query: 用户的建模需求描述。
        sbml_text: 检索到的 SBML 模型文本（伪代码或提取文本）。
        callbacks: 可选的 LangChain 回调列表，用于 Token 统计。

    Returns:
        包含 is_reusable、reuse_reason、nodes、edges 的字典。
    """
    structured_llm = llm.with_structured_output(SBMLParseOutput)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SBML_PARSER_PROMPT),
            ("human", "请分析上述 SBML 模型。"),
        ]
    )
    chain = prompt.partial(
        user_query=user_query,
        retrieved_sbml_text=sbml_text,
    ) | structured_llm
    invoke_kwargs: dict = {}
    if callbacks:
        invoke_kwargs["config"] = {"callbacks": callbacks}
    result: SBMLParseOutput = chain.invoke({}, **invoke_kwargs)
    return result.model_dump()
