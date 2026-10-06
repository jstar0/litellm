import json
from datetime import datetime
from typing import Final

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.utils import EmbeddingResponse, ModelResponse

_PREDICT_ROUTE = "/v1/projects/p/locations/us-central1/publishers/google/models/text-embedding-004:predict"
_INTERACTIONS_ROUTE = "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/interactions"


def _handle(url_route: str, payload: object) -> PassThroughEndpointLoggingTypedDict:
    logging_obj = Logging(
        model="unknown",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.optional_params = {}
    response = httpx.Response(200, json=payload)
    return VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route=url_route,
        result=response.text,
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
        cache_hit=False,
        request_body={"model": "gemini-omni-flash-preview"},
    )


def test_predict_response_with_text_embeddings_is_logged_as_an_embedding_response():
    result = _handle(
        _PREDICT_ROUTE,
        {
            "predictions": [
                {"embeddings": {"values": [0.1, 0.2], "statistics": {"token_count": 3}}},
                {"embeddings": {"values": [0.3, 0.4], "statistics": {"token_count": 4}}},
            ],
            "metadata": {"billableCharacterCount": 9},
        },
    )

    response = result["result"]
    assert isinstance(response, EmbeddingResponse)
    assert response.data == [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
        {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
    ]
    assert response.usage.prompt_tokens == 7
    assert result["kwargs"]["model"] == "text-embedding-004"
    assert result["kwargs"]["custom_llm_provider"] == "vertex_ai"


@pytest.mark.parametrize("payload", [["not", "an", "object"], "text", 7, 1.5, True])
def test_predict_response_that_is_not_a_json_object_is_rejected_without_echoing_it(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        _handle(_PREDICT_ROUTE, payload)

    assert "input_value" not in str(exc_info.value)


def test_interactions_usage_object_is_read_into_prompt_and_completion_tokens():
    result = _handle(
        _INTERACTIONS_ROUTE,
        {
            "id": "interactions/abc",
            "model": "gemini-omni-flash-preview",
            "usage": {
                "total_tokens": 41,
                "total_input_tokens": 12,
                "input_tokens_by_modality": [{"modality": "text", "tokens": 12}],
                "total_output_tokens": 9,
                "output_tokens_by_modality": [{"modality": "text", "tokens": 9}],
                "total_thought_tokens": 20,
            },
        },
    )

    response = result["result"]
    assert isinstance(response, ModelResponse)
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 29
    assert response.usage.completion_tokens_details.text_tokens == 9
    assert result["kwargs"]["custom_llm_provider"] == "vertex_ai"


def test_streamed_vertex_logging_uses_the_regional_response_cost() -> None:
    model: Final = "gemini-3.1-flash-lite"
    start_time: Final = datetime(2026, 1, 1)
    logging_obj: Final = Logging(
        model=model,
        messages=[{"role": "user", "content": "regional cost control"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=start_time,
        litellm_call_id="call-regional-cost",
        function_id="test-vertex-stream",
    )
    logging_obj.custom_llm_provider = "vertex_ai"
    logging_obj.model_call_details["custom_llm_provider"] = "vertex_ai"
    logging_obj.optional_params = {}
    chunks: Final = [
        "data: "
        + json.dumps(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "pong"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 7,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 8,
                },
                "modelVersion": model,
            }
        )
    ]

    result: Final = VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks(
        litellm_logging_obj=logging_obj,
        passthrough_success_handler_obj=PassThroughEndpointLogging(),
        url_route=(
            "https://aiplatform.us.rep.googleapis.com/v1/projects/p/locations/us/"
            f"publishers/google/models/{model}:streamGenerateContent"
        ),
        request_body={},
        endpoint_type=EndpointType.VERTEX_AI,
        start_time=start_time,
        all_chunks=chunks,
        model=model,
        end_time=start_time,
    )
    response: Final = result["result"]
    assert isinstance(response, ModelResponse)
    expected_cost: Final = litellm.completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="vertex_ai",
        vertex_location="us",
    )
    global_cost: Final = litellm.completion_cost(
        completion_response=response,
        model=model,
        custom_llm_provider="vertex_ai",
        vertex_location="global",
    )

    assert expected_cost > global_cost, (expected_cost, global_cost)
    assert result["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    logged_cost: Final = logging_obj._response_cost_calculator(result=response, litellm_model_name=model)
    assert logged_cost == pytest.approx(expected_cost), (logged_cost, expected_cost, response)
