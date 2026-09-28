"""Claude adapter using the System One adapter against OpenRouter."""

from system_one_adapter import AsyncSystemOneAdapterClient
from system_one_adapter.providers.openai import AsyncOpenAIProvider
from typesafe_sdk import RetryPolicy

from jev_experiment.models import Ownership, Prediction, Usage
from jev_experiment.providers.contract import QUESTION_ID, decision_state, ownership_questions

OPENROUTER_CHAT_BASE_URL = "https://openrouter.ai/api/v1"


async def classify_with_claude(
    *,
    provider_name: str,
    case_text: str,
    policy_text: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
    max_retries: int,
) -> Prediction:
    """Classify one transcript with a Claude model through OpenRouter."""

    retry = RetryPolicy(max_retries=max_retries, timeout=timeout_seconds)
    provider = AsyncOpenAIProvider(
        model,
        base_url=OPENROUTER_CHAT_BASE_URL,
        api_key=api_key,
        api="chat_completions",
    )
    async with AsyncSystemOneAdapterClient(
        structured_outputs=True,
        llm_answer_mode="probabilities",
        normalize_probabilities=True,
        n_retry_malformed_structure=1,
        retry=retry,
        model=provider,
    ) as client:
        response = await client.system_one(
            state=decision_state(case_text, policy_text),
            questions=ownership_questions(),
        )

    answer = response.choices[QUESTION_ID]
    reported_costs = []
    served_model = response.model
    for attempt in response.debug.get("llm_attempts", []):
        llm_response = attempt.get("llm_response")
        if not isinstance(llm_response, dict):
            continue
        if isinstance(llm_response.get("model"), str):
            served_model = llm_response["model"]
        raw_usage = llm_response.get("usage")
        if isinstance(raw_usage, dict) and isinstance(raw_usage.get("cost"), (int, float)):
            reported_costs.append(float(raw_usage["cost"]))
    return Prediction(
        provider=provider_name,
        requested_model=model,
        served_model=served_model,
        label=Ownership(answer.choice),
        probabilities={Ownership(key): value for key, value in answer.probabilities.items()},
        confidence=answer.confidence,
        latency_ms=response.usage.latency * 1000,
        retry_count=response.usage.n_retries + response.usage.n_retries_malformed_structure,
        usage=Usage(
            input_tokens=response.usage.input_tokens_total,
            output_tokens=response.usage.output_tokens_total,
            cost_usd=sum(reported_costs) if reported_costs else None,
        ),
        raw_response=response.model_dump(mode="json"),
    )
