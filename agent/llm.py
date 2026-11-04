"""The one place the chat model is called.

ChatLiteLLM keeps LiteLLM as the gateway (switching to Azure only changes MODEL and credentials) and is also a
LangChain model, so Langfuse's callback records every call as a generation without extra code.
"""
import time

import litellm
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel

from agent import config, turn


def get_llm() -> ChatLiteLLM:
    return ChatLiteLLM(model=config.MODEL, request_timeout=config.MODEL_TIMEOUT_S, max_retries=2)


def _cost(input_tokens: int, output_tokens: int) -> float | None:
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=config.MODEL, prompt_tokens=input_tokens, completion_tokens=output_tokens)
        return prompt_cost + completion_cost
    except Exception:
        price = config.FALLBACK_PRICES.get(config.MODEL.split("/")[-1])
        if not price:
            return None
        return input_tokens * price["input"] + output_tokens * price["output"]


def structured(schema: type[BaseModel], system: str, user: str) -> BaseModel:
    """Call the model and return a validated `schema` object. Raises if the model's output doesn't fit.

    Each schema has a `step` name, which becomes this generation's name in the trace.
    """
    start = time.perf_counter()
    message = get_llm().bind(response_format=schema).invoke(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        config={"run_name": schema.step})
    usage = message.usage_metadata or {}
    input_tokens, output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    turn.current().model_calls.append({
        "latency_ms": round((time.perf_counter() - start) * 1000),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": _cost(input_tokens, output_tokens),
    })
    return schema.model_validate_json(message.content)
