# BioDynamics Agent - Token 使用量统计
# 通过 LangChain 回调捕获 LLM 调用产生的 prompt / completion tokens，
# 并在各节点之间累加。

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class UsageAccumulator(BaseCallbackHandler):
    """累加单次 LLM 调用产生的 Token 使用量。"""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """从 LLMResult 中解析 Token 使用量。"""
        try:
            # OpenAI 官方 API 通常通过 llm_output.token_usage 返回
            if response.llm_output and isinstance(response.llm_output, dict):
                token_usage = response.llm_output.get("token_usage") or {}
                if token_usage:
                    self.prompt_tokens += int(token_usage.get("prompt_tokens", 0))
                    self.completion_tokens += int(token_usage.get("completion_tokens", 0))
                    return

            # 部分模型在 generation.message.usage_metadata 中返回
            generation = response.generations[0][0]
            message = getattr(generation, "message", None)
            usage_metadata = getattr(message, "usage_metadata", None) or {}
            if usage_metadata:
                self.prompt_tokens += int(usage_metadata.get("input_tokens", 0))
                self.completion_tokens += int(usage_metadata.get("output_tokens", 0))
        except Exception:
            # Token 统计为辅助功能，解析失败不应中断主流程
            pass


def merge_usage(
    current: dict[str, int] | None,
    delta: dict[str, int] | None,
) -> dict[str, int]:
    """将当前累计用量与本次增量合并，返回新的累计用量字典。"""
    current = current or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    delta = delta or {"prompt_tokens": 0, "completion_tokens": 0}
    prompt = current.get("prompt_tokens", 0) + delta.get("prompt_tokens", 0)
    completion = current.get("completion_tokens", 0) + delta.get("completion_tokens", 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def usage_from_accumulator(accumulator: UsageAccumulator) -> dict[str, int]:
    """从累加器提取本次调用的 Token 用量。"""
    return {
        "prompt_tokens": accumulator.prompt_tokens,
        "completion_tokens": accumulator.completion_tokens,
        "total_tokens": accumulator.prompt_tokens + accumulator.completion_tokens,
    }
