import logging
import os
import time
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from agente_ai import config

logger = logging.getLogger(__name__)

_client: genai.Client | None = None

T = TypeVar("T", bound=BaseModel)


def get_client() -> genai.Client:
    """Reads GEMINI_API_KEY lazily (not at module-import time) so it works
    regardless of whether load_dotenv() ran before or after this module was
    first imported."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (check your .env file)")
        _client = genai.Client(api_key=api_key)
    return _client


def call_with_retry(fn, *args, max_retries: int = 3, base_delay: float = 1.0, **kwargs):
    """Retries fn with exponential backoff — the Gemini free tier rate-limits
    aggressively and the sibling trading_bot_btc project has no retry handling
    at all, which this project deliberately fixes."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "Gemini call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)


def generate_text(
    prompt: str,
    system_instruction: str | None = None,
    max_output_tokens: int = config.MAX_OUTPUT_TOKENS,
) -> str:
    client = get_client()
    gen_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
    )
    resp = call_with_retry(
        client.models.generate_content,
        model=config.GENERATION_MODEL,
        contents=prompt,
        config=gen_config,
    )
    return resp.text


def generate_structured(
    prompt: str,
    response_schema: type[T],
    system_instruction: str | None = None,
    max_output_tokens: int = config.MAX_OUTPUT_TOKENS,
) -> T:
    """Like generate_text but constrains the model to return JSON matching a
    pydantic schema, returned already parsed. Used by nodes that need a typed
    decision (router, critic) rather than free text."""
    client = get_client()
    gen_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=response_schema,
    )
    resp = call_with_retry(
        client.models.generate_content,
        model=config.GENERATION_MODEL,
        contents=prompt,
        config=gen_config,
    )
    return resp.parsed


def embed_text(text: str) -> list[float]:
    client = get_client()
    resp = call_with_retry(
        client.models.embed_content, model=config.EMBEDDING_MODEL, contents=text
    )
    return resp.embeddings[0].values
