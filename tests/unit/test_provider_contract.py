from jev_experiment.providers.contract import (
    OWNERSHIP_CRITERIA,
    QUESTION_ID,
    decision_state,
    ownership_questions,
)


def test_model_facing_state_contains_only_policy_and_transcript() -> None:
    state = decision_state("case body", "policy body")

    assert state == {
        "ownership_policy": "policy body",
        "case_transcript": "case body",
    }
    assert "gold" not in repr(state).lower()


def test_every_provider_uses_the_three_label_choice_contract() -> None:
    question = ownership_questions()[QUESTION_ID]

    assert question.criteria == OWNERSHIP_CRITERIA
    assert set(question.criteria) == {"MINE", "NOT_MINE", "UNSURE"}
