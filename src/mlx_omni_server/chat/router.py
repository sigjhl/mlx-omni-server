import json
from typing import Generator, Optional

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse, StreamingResponse
import os

from .mlx.models import load_model
from .schema import ChatCompletionRequest, ChatCompletionResponse
from .text_models import BaseTextModel
from .mlx.prompt_cache_pool import PromptCachePool

router = APIRouter(tags=["chat—completions"])

_cache_capacity = int(os.getenv("MLX_OMNI_PROMPT_CACHE_CAPACITY", "8"))
_prompt_cache_pool = PromptCachePool(capacity=_cache_capacity)

@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """Create a chat completion"""
    # determine conversation key (header or field in JSON) or fallback
    session_key = (
        x_session_id
        or request.get_extra_params().get("conversation_id")
        or "default"
    )

    # fetch per-session PromptCache and inject into model
    cache = _prompt_cache_pool.get(session_key)
    text_model = _create_text_model(
        request.model, request.get_extra_params().get("adapter_path")
    )
    text_model._prompt_cache = cache

    try:
        if not request.stream:
            completion = text_model.generate(request)
            return JSONResponse(content=completion.model_dump(exclude_none=True))

        async def event_generator() -> Generator[str, None, None]:
            for chunk in text_model.stream_generate(request):
                yield f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    finally:
        # refresh LRU order
        _prompt_cache_pool.touch(session_key)

_last_model_id = None
_last_text_model = None


def _create_text_model(model_id: str, adapter_path: str = None) -> BaseTextModel:
    global _last_model_id, _last_text_model
    if model_id == _last_model_id:
        return _last_text_model

    model = load_model(model_id, adapter_path)
    _last_text_model = model
    _last_model_id = model_id
    return model
