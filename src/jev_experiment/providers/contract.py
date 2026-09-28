"""The model-facing decision contract shared by every provider."""

from pathlib import Path

from typesafe_sdk import Choice

POLICY_PATH = Path("config/ownership_policy.yaml")
QUESTION_ID = "ownership"

OWNERSHIP_CRITERIA = {
    "MINE": "Platform Functionality L3 owns the next concrete action.",
    "NOT_MINE": "Data and Methodology L3 owns the next concrete action.",
    "UNSURE": "The available evidence does not establish which team should act next.",
}


def load_policy_text(path: Path = POLICY_PATH) -> str:
    """Load the versioned policy verbatim for a stable model-facing input."""

    return path.read_text(encoding="utf-8")


def decision_state(case_text: str, policy_text: str) -> dict[str, str]:
    """Build the complete model input without evaluation-only fields."""

    return {
        "ownership_policy": policy_text,
        "case_transcript": case_text,
    }


def ownership_questions() -> dict[str, Choice]:
    """Return a fresh typed question mapping for one provider request."""

    return {
        QUESTION_ID: Choice(
            instructions=(
                "Using the ownership policy and the latest credible information, "
                "which team owns the next concrete action?"
            ),
            criteria=OWNERSHIP_CRITERIA,
        )
    }
