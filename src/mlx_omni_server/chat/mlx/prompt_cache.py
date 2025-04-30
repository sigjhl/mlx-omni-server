"""
Prompt Cache Management Module

This module provides functionality for managing and optimizing model prompt caching,
to improve performance in multi-turn conversations.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import Any, List, Tuple

from ...utils.logger import logger
from mlx_lm.models.cache import KVCache # Make sure this is accessible

@dataclass
class PromptCache:
    """
    Prompt cache class for storing and managing model prompt caches

    Attributes:
        tokens: Cached token sequence (mirrors KV cache state, including hidden tokens)
        cache: Model's KV cache state, a list matching the number of model layers
               Typically List[Tuple[mx.array, mx.array]] where arrays have shape
               (batch, seq_len, num_heads, head_dim) or similar.
        model_key: Model identifier to ensure cache matches the model
    """
    tokens: List[int] = field(default_factory=list)
    cache: List[Any] = field(default_factory=list) # Expect List[Tuple[mx.array, mx.array]]
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

def _truncate_kv_cache(cache_list: List[Any], length: int):
    """
    Truncates the logical length of KV caches in the list to 'length' IN-PLACE,
    preferring the object's `trim` method if available.

    Args:
        cache_list: The list of cache objects (e.g., KVCache instances).
        length: The target logical sequence length to truncate to.
    """
    if not cache_list or length < 0:
        logger.warning(f"Attempted to truncate cache with invalid length {length} or empty list.")
        return

    modified_count = 0
    total_layers = len(cache_list)
    for idx, cache_obj in enumerate(cache_list):
        try:
            current_len = -1
            # Check if it has an offset attribute
            if hasattr(cache_obj, 'offset'):
                current_len = cache_obj.offset
            else:
                logger.warning(f"Cache layer {idx}/{total_layers}: Item lacks 'offset'. Skipping. Type: {type(cache_obj)}")
                continue

            if current_len > length:
                # Calculate amount to trim
                trim_amount = current_len - length
                if trim_amount <= 0:
                     logger.warning(f"Cache layer {idx}/{total_layers}: Calculated trim_amount <= 0 ({trim_amount}) despite current_len ({current_len}) > length ({length}). Skipping trim.")
                     continue # Skip to next layer

                trimmed_successfully = False
                # Try using the trim method first
                if hasattr(cache_obj, 'trim') and callable(cache_obj.trim):
                    try:
                        # Call trim to remove tokens from the end of the logical sequence
                        returned_trim = cache_obj.trim(trim_amount)

                        # Verify if trim reported success and the new offset is correct
                        # KVCache.trim returns the amount actually trimmed.
                        if returned_trim == trim_amount:
                            if cache_obj.offset == length:
                                logger.debug(f"Cache layer {idx}/{total_layers}: Used trim() successfully. Offset reduced by {trim_amount} from {current_len} to {length}.")
                                trimmed_successfully = True
                                modified_count += 1
                            else:
                                # Offset didn't end up as expected, even if trim returned correct amount
                                logger.warning(f"Cache layer {idx}/{total_layers}: trim({trim_amount}) returned {returned_trim}, but final offset is {cache_obj.offset} (expected {length}). State might be inconsistent.")
                                # Attempt to force offset? Risky. Let's just log for now.
                                trimmed_successfully = True # Consider it handled by trim, even if imperfectly
                                modified_count += 1
                        else:
                            # trim() returned a different amount than requested
                            new_offset = current_len - returned_trim
                            logger.warning(f"Cache layer {idx}/{total_layers}: trim({trim_amount}) returned {returned_trim}. Offset is now {new_offset}. Attempting to proceed.")
                            # If it trimmed *something*, maybe it's okay?
                            if returned_trim > 0 :
                                trimmed_successfully = True # Let's assume partial success is better than manual slicing for now
                                modified_count += 1
                            # If returned_trim is 0 or less, trim failed.

                    except Exception as trim_err:
                        logger.warning(f"Cache layer {idx}/{total_layers}: Error calling trim({trim_amount}): {trim_err}. Trying manual slice if possible.")
                        trimmed_successfully = False # Ensure fallback is attempted

                # Fallback: Manual slicing (use only if trim failed or wasn't available)
                if not trimmed_successfully:
                    logger.warning(f"Cache layer {idx}/{total_layers}: trim() failed or unavailable. Attempting manual slice (may cause issues).")
                    if hasattr(cache_obj, 'keys') and hasattr(cache_obj, 'values') and \
                       isinstance(getattr(cache_obj, 'keys', None), mx.array) and \
                       isinstance(getattr(cache_obj, 'values', None), mx.array):

                        physical_key_len = cache_obj.keys.shape[1] if cache_obj.keys is not None else -1
                        physical_val_len = cache_obj.values.shape[1] if cache_obj.values is not None else -1

                        # Log the critical inconsistency if it exists *before* slicing
                        if current_len > physical_key_len and physical_key_len != -1:
                             logger.error(f"INCONSISTENCY PRE-SLICE (Layer {idx}): Offset ({current_len}) > Physical Key Shape ({physical_key_len})")
                        if current_len > physical_val_len and physical_val_len != -1:
                             logger.error(f"INCONSISTENCY PRE-SLICE (Layer {idx}): Offset ({current_len}) > Physical Value Shape ({physical_val_len})")

                        # Perform slicing cautiously
                        target_slice_len_k = min(length, physical_key_len) if physical_key_len != -1 else length
                        if cache_obj.keys is not None and physical_key_len > target_slice_len_k:
                             logger.debug(f"Cache layer {idx}/{total_layers}: Slicing keys from {physical_key_len} to {target_slice_len_k}")
                             cache_obj.keys = cache_obj.keys[:, :target_slice_len_k, :, :]
                        elif cache_obj.keys is not None:
                             logger.debug(f"Cache layer {idx}/{total_layers}: Key slice not needed (target {length}, physical {physical_key_len})")


                        target_slice_len_v = min(length, physical_val_len) if physical_val_len != -1 else length
                        if cache_obj.values is not None and physical_val_len > target_slice_len_v:
                             logger.debug(f"Cache layer {idx}/{total_layers}: Slicing values from {physical_val_len} to {target_slice_len_v}")
                             cache_obj.values = cache_obj.values[:, :target_slice_len_v, :, :]
                        elif cache_obj.values is not None:
                             logger.debug(f"Cache layer {idx}/{total_layers}: Value slice not needed (target {length}, physical {physical_val_len})")


                        # Force the offset to the target length after slicing attempt
                        logger.debug(f"Cache layer {idx}/{total_layers}: Manually setting offset to {length} after slice attempt.")
                        cache_obj.offset = length
                        modified_count += 1
                    else:
                        logger.error(f"Cache layer {idx}/{total_layers}: Cannot perform manual slice - KVCache structure not recognized or tensors missing.")


            elif current_len < length:
                logger.warning(f"Cache layer {idx}/{total_layers}: Offset ({current_len}) already <= target length ({length}). No truncation needed.")
            # else: current_len == length, no action needed

        except Exception as e:
            # Log unexpected errors during the processing of a single layer
            logger.error(f"Error processing cache layer {idx}/{total_layers} during truncation (target length {length}): {e}. Skipping layer.", exc_info=True)

    if modified_count > 0:
        logger.info(f"Cache truncation attempted for {modified_count}/{total_layers} layers to target length {length}.")
    else:
        logger.debug(f"No cache layers required truncation to target length {length}.")

def process_prompt_cache(
    prompt: List[int], prompt_cache: PromptCache, model_key: str, model: Any
) -> Tuple[List[int], int]:
    """
    Process prompt cache, truncating KV cache if new prompt diverges
    or is shorter than the cached state after a common prefix.

    Args:
        prompt: List of encoded prompt tokens from the *current* request (user-visible history based).
        prompt_cache: Prompt cache object holding state from *previous* turns (inc. hidden tokens).
        model_key: Model identifier.
        model: Model object used to create/reset cache.

    Returns:
        Tuple[List[int], int]: Tuple containing:
            1. List of prompt tokens to process next.
            2. Number of tokens reused from the cache (common prefix length).
    """
    from mlx_lm.models.cache import make_prompt_cache # Lazy import

    cache_tokens = prompt_cache.tokens
    cache_len    = len(cache_tokens)
    prompt_len   = len(prompt)

    # --- Safety Check 1: Model Change ---
    if prompt_cache.model_key != model_key:
        logger.debug(f"Model key changed ('{prompt_cache.model_key}' -> '{model_key}'), resetting cache.")
        prompt_cache.model_key = model_key
        prompt_cache.cache     = make_prompt_cache(model)
        prompt_cache.tokens    = prompt.copy()
        return prompt, 0

    # --- Find Common Prefix ---
    common_len = 0
    for i in range(min(cache_len, prompt_len)):
        if cache_tokens[i] != prompt[i]:
            break
        common_len += 1

    # --- Cache Handling ---
    if common_len == 0:
        # No overlap - full reset required
        logger.debug("No common prefix found with cache. Resetting.")
        prompt_cache.cache  = make_prompt_cache(model)
        prompt_cache.tokens = prompt.copy()
        return prompt, 0
    else:
        # There is a common prefix of length common_len.
        # The KV cache state up to common_len is potentially reusable.

        if common_len < cache_len:
            # The new prompt diverges *or* is shorter than the full cached sequence.
            # We MUST truncate the KV cache to the length of the common prefix
            # to discard the invalid state beyond that point.
            logger.debug(
                f"Common prefix length ({common_len}) is less than cache token length ({cache_len}). "
                f"Truncating KV cache state."
            )
            _truncate_kv_cache(prompt_cache.cache, common_len)

        # If common_len == cache_len, the cache is either identical or a prefix
        # of the new prompt. No truncation needed in this case.

        # Update the token mirror to reflect the *target* state (the new full prompt)
        # This is important so that the *next* call to process_prompt_cache compares against this new state.
        prompt_cache.tokens = prompt.copy()

        if common_len == prompt_len:
            # New prompt is identical to the (potentially truncated) cache state.
            logger.debug(f"New prompt matches cached state up to length {common_len}. No new tokens to process.")
            return [], common_len
        else:
            # New prompt extends the (potentially truncated) cache state.
            logger.debug(f"Reusing cache prefix of length {common_len}. Processing {prompt_len - common_len} new tokens.")
            # Return only the tokens *after* the common prefix.
            return prompt[common_len:], common_len