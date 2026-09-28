from dataclasses import dataclass

from jev_experiment.providers.jev import _raw_response_record, _reported_cost


@dataclass
class FakeHttpResponse:
    payload: dict[str, object]
    status_code: int = 200

    @property
    def headers(self) -> dict[str, str]:
        return {
            "x-generation-id": "gen-test",
            "x-typesafe-request-id": "req-test",
        }

    def json(self) -> dict[str, object]:
        return self.payload


class FakeResponse:
    raw_http_response = FakeHttpResponse(
        {"usage": {"input_tokens": 100, "output_tokens": 5, "cost": 0.0000042}}
    )

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {"usage": {"input_tokens": 100, "output_tokens": 5}}


def test_reported_cost_reads_raw_openrouter_usage() -> None:
    assert _reported_cost({"usage": {"cost": 0.0000042}}) == 0.0000042
    assert _reported_cost({"usage": {"input_tokens": 100}}) is None


def test_raw_response_record_preserves_cost_and_generation_metadata() -> None:
    record, cost = _raw_response_record(FakeResponse())

    assert cost == 0.0000042
    assert record["body"] == FakeResponse.raw_http_response.payload
    assert record["metadata"] == {
        "status_code": 200,
        "generation_id": "gen-test",
        "request_id": "req-test",
    }
    assert record["normalized"] == {
        "usage": {"input_tokens": 100, "output_tokens": 5}
    }
