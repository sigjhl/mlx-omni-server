# --- START OF FILE mlx_model.py ---

import time
import uuid
from typing import Any, Dict, Generator, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import GenerationResponse, stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from ...utils.logger import logger
from ..schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    PromptTokensDetails, # Import added for usage reporting
    Role,
)
from ..text_models import BaseTextModel, GenerateResult
from .outlines_logits_processor import OutlinesLogitsProcessor
from .prompt_cache import PromptCache, process_prompt_cache # Assuming update_prompt_cache is not directly needed here
from .stop_tokens_checker import StopTokensChecker
from .tools.chat_tokenizer import ChatTokenizer


class MLXModel(BaseTextModel):
    """MLX Chat Model wrapper with internal parameter management"""

    def __init__(self, model_id: str, model: nn.Module, tokenizer: ChatTokenizer):
        self._model_id = model_id
        self._model: nn.Module = model
        self._default_max_tokens = 2048
        self._default_temperature = 1.0
        self._default_top_p = 1.0
        self._default_top_k = -1
        self._chat_tokenizer = tokenizer
        self._prompt_cache = PromptCache() # Instance specific cache state
        self._cached_token_count = (
            0  # Track the number of cached tokens used *for the input* in the current request
        )
        logger.info(f"Initialized MLXModel with model_id: {model_id}")

    def _get_generation_params(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Extracts additional generation parameters from the request."""
        params = request.get_extra_params()
        # Filter out known parameters handled elsewhere or specific to other models
        known_params = {
            "top_k",
            "min_tokens_to_keep",
            "min_p",
            "adapter_path",
            # Parameters handled directly by FastAPI/Pydantic or standard OpenAI fields
            "model", "messages", "temperature", "top_p", "max_tokens", "max_completion_tokens",
            "stream", "stream_options", "stop", "presence_penalty", "frequency_penalty",
            "logit_bias", "logprobs", "top_logprobs", "n", "user", "response_format",
            "tools", "tool_choice", "seed",
        }
        # Allow repetition_penalty, but handle presence_penalty separately for logits processor
        if "repetition_penalty" in params and not request.presence_penalty:
             # If presence_penalty is set, let make_logits_processors handle it.
             # Otherwise, pass repetition_penalty through if it exists.
             pass
        else:
             known_params.add("repetition_penalty") # Don't pass if presence_penalty is used

        return {k: v for k, v in params.items() if k not in known_params}

    def _process_logprobs(
        self,
        tokenizer: TokenizerWrapper,
        response: GenerationResponse,
        top_k: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Process logprobs information from generation response to match OpenAI format"""
        if not response.logprobs or top_k == 0: # Check if logprobs exist and top_k isn't explicitly 0
             return None

        current_token = response.token
        current_logprobs = response.logprobs

        # Get current token info
        token_str = tokenizer.decode([current_token])
        # Handle potential decoding errors or empty strings
        if not token_str:
            token_str = f"<ID:{current_token}>" # Fallback representation
        token_logprob = current_logprobs[current_token].item()
        try:
            token_bytes = list(token_str.encode("utf-8"))
        except UnicodeEncodeError:
            token_bytes = list(f"<Bytes:{current_token}>".encode("utf-8")) # Fallback bytes

        # Base token info
        token_info = {
            "token": token_str,
            "logprob": token_logprob,
            "bytes": token_bytes,
        }

        # Process top logprobs if requested (top_k > 0)
        top_logprobs_list = []
        if top_k is not None and top_k > 0:
            # Clamp top_k to the vocab size
            top_k = min(top_k, len(current_logprobs))
            # Get indices and values of top_k tokens
            top_indices = mx.argpartition(-current_logprobs, kth=top_k - 1)[:top_k]
            top_probs = current_logprobs[top_indices]

            # Create detailed token information for each top token
            for idx, logprob in zip(top_indices.tolist(), top_probs.tolist()):
                top_token_str = tokenizer.decode([idx])
                if not top_token_str:
                     top_token_str = f"<ID:{idx}>"
                try:
                    top_token_bytes = list(top_token_str.encode("utf-8"))
                except UnicodeEncodeError:
                    top_token_bytes = list(f"<Bytes:{idx}>".encode("utf-8"))

                top_logprobs_list.append(
                    {"token": top_token_str, "logprob": logprob, "bytes": top_token_bytes}
                )

        return {**token_info, "top_logprobs": top_logprobs_list}

    def _stream_generate(
        self,
        prompt_string: str, # Use the raw string prompt for tokenization here
        request: ChatCompletionRequest,
    ) -> Generator[GenerateResult, None, None]:
        """
        Internal generator handling cache processing, model calls, and yielding results.
        """
        try:
            # --- Setup ---
            params = self._get_generation_params(request)
            tokenizer = self._chat_tokenizer.tokenizer
            stop_checker = None
            if request.stop:
                stop_checker = StopTokensChecker(
                    stop_words=request.stop,
                    tokenizer=tokenizer,
                )

            logits_processors = None
            if request.response_format and request.response_format.type == "json_object":
                 # Assuming response_format.schema is handled by Outlines if present
                 # Note: Need to ensure OutlinesLogitsProcessor handles schema correctly
                logits_processors = [
                    OutlinesLogitsProcessor(
                        self._chat_tokenizer.tokenizer, request.response_format
                    )
                ]
            else:
                # Handle presence_penalty (maps to repetition_penalty in mlx-lm)
                # frequency_penalty is not directly supported by mlx-lm's default processors
                repetition_penalty = request.presence_penalty or params.get("repetition_penalty")
                if repetition_penalty:
                    logits_processors = make_logits_processors(repetition_penalty=repetition_penalty)

            # --- Cache Processing ---
            tokenized_prompt = tokenizer.encode(prompt_string) # Tokenize the input string
            processed_prompt_tokens, cached_token_count = process_prompt_cache(
                tokenized_prompt, self._prompt_cache, self._model_id, self._model
            )
            # Store the number of cached tokens used for input processing for later usage reporting
            self._cached_token_count = cached_token_count
            logger.debug(
                f"Cache Processing: Input tokens={len(tokenized_prompt)}, "
                f"Reused cache tokens={cached_token_count}, "
                f"New tokens to process={len(processed_prompt_tokens)}"
            )
            # --- End Cache Processing ---

            # --- Handle Regenerate Case ---
            prompt_to_pass_to_generator: List[int]
            if not processed_prompt_tokens and cached_token_count > 0:
                # Regenerate case: cache matched the entire input prompt.
                # Need to provide the last token of the matched sequence to mlx_lm.generate.
                if cached_token_count > len(tokenized_prompt):
                     # This indicates an internal inconsistency, log and potentially reset
                     logger.error(f"Regenerate Inconsistency: cached_token_count ({cached_token_count}) > len(tokenized_prompt) ({len(tokenized_prompt)}). Resetting cache.")
                     # Perform a full reset as a safety measure
                     from mlx_lm.models.cache import make_prompt_cache
                     self._prompt_cache.cache = make_prompt_cache(self._model)
                     self._prompt_cache.tokens = tokenized_prompt.copy()
                     self._cached_token_count = 0
                     prompt_to_pass_to_generator = tokenized_prompt # Process the whole prompt again
                else:
                    last_cached_token_id = tokenized_prompt[cached_token_count - 1]
                    prompt_to_pass_to_generator = [last_cached_token_id]
                    logger.debug(
                        f"Regenerate case detected. Using last cached token ID {last_cached_token_id} "
                        f"as starting point for mlx_lm.generate."
                    )
            elif not processed_prompt_tokens and cached_token_count == 0:
                 # Input prompt was empty after encoding, and no cache hit.
                 logger.error("Cannot generate from an empty prompt with no cache hit.")
                 raise ValueError("Cannot generate from an empty prompt with no cache hit.")
            else:
                # Normal case (or edit case): process the new tokens.
                prompt_to_pass_to_generator = processed_prompt_tokens
            # --- End Handle Regenerate Case ---

            # --- Generation Setup ---
            current_generation_tokens = [] # Tokens generated *in this specific call*
            last_decoded_text = ""
            prompt_tokens_processed_by_model = len(prompt_to_pass_to_generator) # Track tokens actually fed to model

            max_completion_tokens = (
                request.max_completion_tokens
                or request.max_tokens
                or self._default_max_tokens
            )
            sampler = make_sampler(
                temp=request.temperature if request.temperature is not None else self._default_temperature,
                top_p=request.top_p if request.top_p is not None else self._default_top_p,
                min_p=params.get("min_p", 0.0),
                # min_tokens_to_keep seems less relevant here, handled internally by sampler maybe?
                top_k=params.get("top_k", self._default_top_k),
                # seed=request.seed # Pass seed if supported by make_sampler/underlying generator
            )

            # The prompt_cache object itself contains the KVCache list which mlx_lm expects
            current_cache_state = self._prompt_cache.cache

            # --- Generation Loop ---
            for response in stream_generate(
                model=self._model,
                tokenizer=tokenizer,
                prompt=prompt_to_pass_to_generator, # Use the potentially adjusted prompt list
                max_tokens=max_completion_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=current_cache_state, # Pass the actual KVCache list
                **params, # Pass any extra compatible params
            ):
                # Check for early exit (e.g., error in generator)
                if response.finish_reason == "error":
                     logger.error("Error reported by mlx_lm.stream_generate.")
                     # You might want to yield a specific error result here
                     break # Exit the loop

                # Append the generated token to our internal mirror *and* local list for decoding
                # The KVCache objects in current_cache_state are updated internally by stream_generate
                self._prompt_cache.tokens.append(response.token)
                current_generation_tokens.append(response.token)

                # Process logprobs if requested
                logprobs_detail = None
                if request.logprobs:
                    logprobs_detail = self._process_logprobs(
                        tokenizer, response, request.top_logprobs
                    )

                # Check stop conditions
                finish_reason = response.finish_reason # Can be None, 'stop', 'length', 'error'
                should_trim_stop_word = False
                if request.stop and stop_checker:
                    stop_condition = stop_checker.check_stop_condition(current_generation_tokens)
                    if stop_condition.stop_met:
                        finish_reason = "stop" # Override reason if stop word found
                        if stop_condition.trim_length > 0:
                            # Remove stop word tokens from the end of our local list
                            current_generation_tokens = current_generation_tokens[:-stop_condition.trim_length]
                            # Also remove them from the cache mirror
                            self._prompt_cache.tokens = self._prompt_cache.tokens[:-stop_condition.trim_length]
                            # Note: We cannot easily undo the KV cache update for the trimmed tokens.
                            # This might lead to slight inefficiency if generation continues later.
                            should_trim_stop_word = True
                            logger.debug(f"Stop word found, trimming {stop_condition.trim_length} token(s).")


                # Decode the *cumulative* generated text for this call
                current_full_text = tokenizer.decode(current_generation_tokens)
                # Calculate the delta text since the last yield
                delta_text = current_full_text[len(last_decoded_text):]

                # Yield the result chunk
                # Only yield if there's new text OR if we trimmed (to send the final state)
                if delta_text or should_trim_stop_word or finish_reason:
                    yield GenerateResult(
                        text=delta_text,
                        token=response.token, # The raw token ID for this step
                        finish_reason=finish_reason,
                        # Report prompt tokens processed *by the model* in this call
                        prompt_tokens=prompt_tokens_processed_by_model,
                        # Report cumulative generation tokens *so far in this call*
                        generation_tokens=len(current_generation_tokens),
                        logprobs=logprobs_detail,
                    )
                    last_decoded_text = current_full_text # Update history for delta calculation

                # If generation finished (stop word, length, error), exit the loop
                if finish_reason:
                    break
            # --- End Generation Loop ---

            logger.debug(
                f"Internal stream completed. Cache mirror has {len(self._prompt_cache.tokens)} tokens."
            )

        except Exception as e:
            logger.error(f"Error during internal stream generation: {str(e)}", exc_info=True)
            # Ensure the generator stops cleanly, maybe yield an error result?
            yield GenerateResult(text="", finish_reason="error") # Yield minimal error signal
            raise # Re-raise the exception


    def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Handles non-streaming chat completion requests."""
        try:
            # --- Prepare Prompt ---
            prompt_string = self._chat_tokenizer.encode(
                messages=request.messages,
                tools=request.tools,
                tool_choice=request.tool_choice if request.tool_choice else None,
            )
            logger.debug(f"Encoded prompt string for generate:\n{prompt_string}")

            # --- Accumulate Stream Results ---
            final_completion_text = ""
            logprobs_result_list = []
            final_result: Optional[GenerateResult] = None
            generation_token_count = 0

            for result in self._stream_generate(
                prompt=prompt_string,
                request=request,
            ):
                final_completion_text += result.text
                if request.logprobs and result.logprobs:
                    logprobs_result_list.append(result.logprobs)
                generation_token_count += 1 # Count generated tokens
                final_result = result # Keep track of the last result state

                if result.finish_reason:
                    break

            # Adjust count if last yielded token was empty text but part of generation
            if final_result and not final_result.text and generation_token_count > 0:
                 generation_token_count -=1 # Don't count the final empty yield if any

            if final_result is None:
                # Handle case where _stream_generate yielded nothing (e.g., immediate error)
                logger.error("No generation results received from _stream_generate.")
                raise RuntimeError("Generation failed to produce any tokens.")

            logger.debug(f"Final generated text:\n{final_completion_text}")

            # --- Format Response ---
            message: ChatMessage
            if request.tools:
                 # Attempt to decode potential tool calls from the completion text
                 try:
                      message = self._chat_tokenizer.decode(final_completion_text)
                      # Ensure role is assistant, decode might return user/tool if parsing fails
                      if message.role != Role.ASSISTANT:
                           message = ChatMessage(role=Role.ASSISTANT, content=final_completion_text, tool_calls=message.tool_calls)
                 except Exception as decode_err:
                      logger.warning(f"Failed to decode tool calls from completion: {decode_err}. Returning raw text.")
                      message = ChatMessage(role=Role.ASSISTANT, content=final_completion_text)
            else:
                message = ChatMessage(role=Role.ASSISTANT, content=final_completion_text)

            # Determine final finish reason
            finish_reason = final_result.finish_reason
            if message.tool_calls:
                finish_reason = "tool_calls" # Override if tool calls are present

            # --- Calculate Usage ---
            # Prompt tokens = cached tokens reused + new prompt tokens processed by model
            prompt_tokens_processed_by_model = final_result.prompt_tokens
            cached_tokens = self._cached_token_count # From input processing
            total_prompt_tokens = cached_tokens + prompt_tokens_processed_by_model

            # Completion tokens: use the count tracked during accumulation
            total_completion_tokens = generation_token_count

            logger.debug(
                 f"Generate Usage: cached={cached_tokens}, new_prompt={prompt_tokens_processed_by_model}, "
                 f"generated={total_completion_tokens}"
            )

            prompt_tokens_details = None
            if cached_tokens > 0:
                prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)

            usage = ChatCompletionUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
                prompt_tokens_details=prompt_tokens_details,
            )

            # --- Create Response Object ---
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
                created=int(time.time()),
                model=request.model, # Return the requested model name
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=message,
                        finish_reason=finish_reason,
                        logprobs=(
                            {"content": logprobs_result_list}
                            if logprobs_result_list and request.logprobs # Check request flag again
                            else None
                        ),
                    )
                ],
                usage=usage,
            )
        except Exception as e:
            logger.error(f"Failed to generate completion: {str(e)}", exc_info=True)
            # Consider returning a standard error response format if possible
            raise RuntimeError(f"Failed to generate completion: {str(e)}")


    def stream_generate(
        self,
        request: ChatCompletionRequest,
    ) -> Generator[ChatCompletionChunk, None, None]:
        """
        Handles streaming chat completion requests, yielding chunks.
        Acts as a wrapper around _stream_generate.
        """
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"

            # --- Prepare Prompt String ---
            # We only need the string here; tokenization and cache happens inside _stream_generate
            prompt_string = self._chat_tokenizer.encode(
                messages=request.messages,
                tools=request.tools,
                # tool_choice is handled in non-streaming `generate` or needs logic here if required for streaming start
            )
            logger.debug(f"Encoded prompt string for stream_generate:\n{prompt_string}")

            # --- Iterate through Internal Generator ---
            generation_token_count = 0
            final_prompt_token_count = 0 # Track prompt tokens processed by model
            last_result : Optional[GenerateResult] = None

            for result in self._stream_generate( # Call the internal generator
                prompt_string=prompt_string, # Correct keyword argument name
                request=request,
            ):
                created = int(time.time())
                # Update counts based on the result from the internal generator
                generation_token_count = result.generation_tokens # Use cumulative count from result
                final_prompt_token_count = result.prompt_tokens # Use count from result
                last_result = result # Store last result for final usage calc

                # Create the chunk to yield
                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatMessage(role=Role.ASSISTANT, content=result.text), # Delta contains only new text
                            finish_reason=result.finish_reason,
                            logprobs=result.logprobs, # Logprobs for the current token
                        )
                    ],
                    # Usage is typically only included in the *final* chunk if requested
                    usage=None,
                )

                # If the internal generator signaled completion, break here
                if result.finish_reason:
                    break

            # --- Final Usage Chunk (if requested and enabled) ---
            if request.stream_options and request.stream_options.include_usage:
                 # Calculate final usage based on the state *after* the internal generator finished
                 created = int(time.time())
                 cached_tokens = self._cached_token_count # Fetched from input processing stage

                 # Ensure final_prompt_token_count reflects the actual processing
                 # If generation never started (e.g., error before first yield), use 0
                 prompt_tokens_processed = final_prompt_token_count if last_result else 0

                 # Ensure generation_token_count is accurate
                 # If generation never started, use 0
                 total_completion_tokens = generation_token_count if last_result else 0

                 total_prompt_tokens = cached_tokens + prompt_tokens_processed

                 logger.debug(
                      f"Stream Final Usage: cached={cached_tokens}, new_prompt={prompt_tokens_processed}, "
                      f"generated={total_completion_tokens}"
                 )

                 prompt_tokens_details = None
                 if cached_tokens > 0:
                     prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)

                 # Yield the final chunk containing only usage information
                 yield ChatCompletionChunk(
                     id=chat_id,
                     created=created,
                     model=request.model,
                     choices=[], # Final usage chunk has no delta choice according to OpenAI spec
                     usage=ChatCompletionUsage(
                         prompt_tokens=total_prompt_tokens,
                         completion_tokens=total_completion_tokens,
                         total_tokens=total_prompt_tokens + total_completion_tokens,
                         prompt_tokens_details=prompt_tokens_details,
                     ),
                 )

        except Exception as e:
            logger.error(f"Error during stream_generate wrapper: {str(e)}", exc_info=True)
            # How to signal error via stream? Yielding a chunk with error info might be best.
            # For now, just re-raise to let FastAPI handle it.
            raise

# --- END OF FILE mlx_model.py ---