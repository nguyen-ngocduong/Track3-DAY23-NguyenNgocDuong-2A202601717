"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()


def get_llm(model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. GROQ_API_KEY → ChatGroq
    2. GEMINI_API_KEY → ChatGoogleGenerativeAI
    3. OPENAI_API_KEY → ChatOpenAI
    4. ANTHROPIC_API_KEY → ChatAnthropic

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    if os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-groq") from exc
        return ChatGroq(  # type: ignore[call-arg]
            model=model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),  # type: ignore[arg-type]
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
        )

    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import (  # type: ignore[import-not-found]
                ChatGoogleGenerativeAI,
            )
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        return ChatAnthropic(
            model=model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set GROQ_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, "
        "or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )
