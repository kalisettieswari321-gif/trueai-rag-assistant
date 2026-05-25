"""
LLM service: wraps Google Gemini API with retry logic,
timeout handling, error classification, and token logging.
"""

import os
import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, InvalidArgument, PermissionDenied
from dotenv import load_dotenv
from fastapi import HTTPException
from app.utils.logger import get_logger, log_token_usage

load_dotenv()
logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.2
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0   # seconds (exponential backoff)
REQUEST_TIMEOUT = 30     # seconds

# ── Initialise Gemini ─────────────────────────────────────────────────────────
_api_key = os.getenv("GEMINI_API_KEY")
_gemini_model = None

if _api_key:
    genai.configure(api_key=_api_key)
    _generation_config = genai.types.GenerationConfig(
        temperature=TEMPERATURE,
        max_output_tokens=1024,
    )
    _gemini_model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=_generation_config,
    )
else:
    logger.warning("GEMINI_API_KEY not set in environment. LLM requests will fail with 401.")


# ── Public API ────────────────────────────────────────────────────────────────

def generate(prompt: str, session_id: str = "unknown") -> tuple[str, int]:
    """
    Call Gemini with the given prompt. Retries on rate-limit errors.

    Returns:
        (reply_text, tokens_used)

    Raises:
        HTTPException 401 for invalid API key
        HTTPException 429 for persistent rate limiting
        HTTPException 504 for timeout
        HTTPException 500 for unexpected errors
    """
    if not _gemini_model:
        logger.error(f"LLM | API KEY MISSING | session={session_id}")
        raise HTTPException(
            status_code=401,
            detail="GEMINI_API_KEY is not configured in the environment. Please add it to your .env file."
        )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"LLM | attempt={attempt} | session={session_id} | model={GEMINI_MODEL}")

            response = _gemini_model.generate_content(
                prompt,
                request_options={"timeout": REQUEST_TIMEOUT},
            )

            reply = response.text.strip()

            # Extract token usage from usage_metadata if available
            tokens_used = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                tokens_used = (
                    getattr(meta, "prompt_token_count", 0) +
                    getattr(meta, "candidates_token_count", 0)
                )

            log_token_usage(logger, session_id, tokens_used)
            logger.info(f"LLM | SUCCESS | tokens={tokens_used}")
            return reply, tokens_used

        except PermissionDenied as e:
            logger.error(f"LLM | INVALID API KEY | {e}")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing Gemini API key. Check your .env file."
            )

        except ResourceExhausted as e:
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(f"LLM | RATE LIMIT | attempt={attempt} | retrying in {wait}s | {e}")
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                raise HTTPException(
                    status_code=429,
                    detail="Gemini API rate limit exceeded. Please try again later."
                )

        except InvalidArgument as e:
            logger.error(f"LLM | INVALID ARGUMENT | {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request to Gemini API: {str(e)}"
            )

        except TimeoutError as e:
            logger.error(f"LLM | TIMEOUT | session={session_id} | {e}")
            raise HTTPException(
                status_code=504,
                detail="The LLM request timed out. Please try again."
            )

        except Exception as e:
            logger.error(f"LLM | UNEXPECTED ERROR | attempt={attempt} | {type(e).__name__}: {e}")
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE_DELAY)
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"LLM service error: {str(last_error)}"
                )

    raise HTTPException(status_code=500, detail="LLM service failed after all retries.")
