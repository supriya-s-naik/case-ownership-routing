"""Live provider adapters for the LangGraph workflow."""

from jev_experiment.providers.base import CallableProvider, ClassificationRequest
from jev_experiment.providers.claude import classify_with_claude
from jev_experiment.providers.jev import classify_with_jev


def build_live_providers(
    *,
    api_key: str,
    models: dict[str, str],
    timeout_seconds: float,
) -> tuple[CallableProvider, ...]:
    """Build the three live adapters; graph nodes own whole-call retries."""

    required = {"jev", "haiku", "sonnet"}
    missing = sorted(required - set(models))
    if missing:
        raise ValueError(f"Missing model configuration for: {', '.join(missing)}")

    async def call_jev(request: ClassificationRequest):
        return await classify_with_jev(
            case_text=request.case_transcript,
            policy_text=request.policy_text,
            api_key=api_key,
            model=models["jev"],
            timeout_seconds=timeout_seconds,
            max_retries=0,
        )

    def claude_adapter(provider_name: str) -> CallableProvider:
        async def call_claude(request: ClassificationRequest):
            return await classify_with_claude(
                provider_name=provider_name,
                case_text=request.case_transcript,
                policy_text=request.policy_text,
                api_key=api_key,
                model=models[provider_name],
                timeout_seconds=timeout_seconds,
                max_retries=0,
            )

        return CallableProvider(provider_name, models[provider_name], call_claude)

    return (
        CallableProvider("jev", models["jev"], call_jev),
        claude_adapter("haiku"),
        claude_adapter("sonnet"),
    )
