import logging
import sys
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with structured console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def log_retrieval(logger: logging.Logger, query: str, results: list):
    """Log similarity search results with scores."""
    logger.info(f"RETRIEVAL | query='{query[:60]}...' | results={len(results)}")
    for r in results:
        logger.info(
            f"  chunk_id={r['chunk_id']} | title='{r['title']}' | "
            f"score={r['score']:.4f}"
        )


def log_token_usage(logger: logging.Logger, session_id: str, tokens_used: int):
    """Log LLM token consumption."""
    logger.info(
        f"TOKENS | session={session_id} | tokens_used={tokens_used} | "
        f"timestamp={datetime.utcnow().isoformat()}"
    )
