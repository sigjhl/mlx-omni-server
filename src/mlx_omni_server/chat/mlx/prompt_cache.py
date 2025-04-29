"""
Prompt Cache Management Module

This module provides functionality for managing and optimizing model prompt caching,
to improve performance in multi-turn conversations.
"""

from dataclasses import dataclass, field
from typing import Any, List, Tuple

from ...utils.logger import logger


@dataclass
class PromptCache:
    """
    Prompt cache class for storing and managing model prompt caches

    Attributes:
        tokens: Cached token sequence
        cache: Model's KV cache state, a list matching the number of model layers
        model_key: Model identifier to ensure cache matches the model
    """

    tokens: List[int] = field(default_factory=list)
    cache: List[Any] = field(default_factory=list)
    model_key: str = ""


def update_prompt_cache(
    prompt_cache: PromptCache,
    tokenized_prompt: List[int],
    model_key: str,
    cache_state: List[Any] = None,
) -> None:
    """
    Update prompt cache

    Args:
        prompt_cache: Prompt cache object
        tokenized_prompt: List of encoded prompt tokens
        model_key: Model identifier
        cache_state: Model's KV cache state, updates cache if provided
    """
    prompt_cache.tokens = tokenized_prompt.copy()
    prompt_cache.model_key = model_key

    # Update cache if new cache state is provided
    if cache_state is not None:
        prompt_cache.cache = cache_state
        logger.debug(f"Updated cache state with {len(cache_state)} layers")

    logger.debug(f"Updated cache with {len(tokenized_prompt)} tokens")


def process_prompt_cache(
    prompt: List[int], prompt_cache: PromptCache, model_key: str, model: Any
) -> Tuple[List[int], int]:
    """
    Process prompt cache using official logic

    Args:
        prompt: List of encoded prompt tokens
        prompt_cache: Prompt cache object
        model_key: Model identifier
        model: Model object used to create cache

    Returns:
        Tuple[List[int], int]: Tuple containing:
            1. List of prompt tokens to process (if cached, only returns uncached portion)
            2. Number of tokens retrieved from cache
    """
    from mlx_lm.models.cache import make_prompt_cache

    cache_tokens = prompt_cache.tokens
    cache_len    = len(cache_tokens)
    prompt_len   = len(prompt)

    # Full reset if model changed
    if prompt_cache.model_key != model_key:
        prompt_cache.model_key = model_key
        prompt_cache.cache     = make_prompt_cache(model)
        prompt_cache.tokens    = prompt.copy()
        return prompt, 0

    # Longest common prefix
    common_len = 0
    for a, b in zip(cache_tokens, prompt):
        if a != b:
            break
        common_len += 1

    if common_len == 0:
        # No overlap – reset
        prompt_cache.cache  = make_prompt_cache(model)
        prompt_cache.tokens = prompt.copy()
        return prompt, 0

    # Reuse the prefix that matches
    prompt_cache.tokens = prompt.copy()            # keep mirror in sync
    return prompt[common_len:], common_len
