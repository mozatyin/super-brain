"""Retry wrapper for Anthropic/OpenRouter API calls (handles transient errors)."""

from __future__ import annotations

import time

import anthropic


def retry_api_call(fn, max_retries=3, base_delay=5):
    """Retry API calls on transient errors (403, 429, 500+)."""
    for attempt in range(max_retries):
        try:
            return fn()
        except (anthropic.PermissionDeniedError, anthropic.RateLimitError,
                anthropic.InternalServerError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"    [retry {attempt+1}/{max_retries}] {type(e).__name__}, waiting {delay}s...")
            time.sleep(delay)
    raise RuntimeError("Unreachable")
