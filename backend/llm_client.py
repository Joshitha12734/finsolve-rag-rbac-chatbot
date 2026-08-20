"""
Unified LLM client — every place in this app that calls an LLM (answer
generation, NL-to-SQL, query classification, reranking, evaluation)
imports `chat()` from here instead of talking to a provider SDK directly.

Why this exists: the app originally called the Anthropic SDK directly in
five different files. Centralizing it means switching providers is a
one-file change instead of a five-file find-and-replace, and it's the
kind of provider abstraction a real production system would want anyway.

Providers:
  - "groq" (default) — fast inference, generous free tier, OpenAI-compatible
    SDK. Set GROQ_API_KEY. Model default: llama-3.3-70b-versatile.
  - "anthropic" — Claude models, paid. Set ANTHROPIC_API_KEY and
    LLM_PROVIDER=anthropic.
  - "ollama" — fully local, no API key, no internet dependency once a
    model is pulled. Set LLM_PROVIDER=ollama, run `ollama pull <model>`,
    and make sure `ollama serve` is running (it runs automatically after
    install on most platforms). Model default: llama3.2. Unlike the API
    providers, "configured" here means "the local Ollama server is
    actually reachable" (checked live, with a short timeout) rather than
    "a key is set" — there's no key to set.

Every call site still degrades gracefully (falls back to raw retrieved
context, skips reranking, uses heuristic classification, etc.) when the
active provider isn't usable — see api_key_configured().
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_ollama_reachable_cache: bool | None = None  # cached per-process; see _ollama_reachable()


def api_key_configured() -> bool:
    """Despite the name (kept for the API providers this was originally
    written for), for Ollama this means 'is the local server reachable',
    not 'is a key set' — there is no key. Callers just need a single
    yes/no on 'can I actually generate text right now'."""
    if PROVIDER == "groq":
        return bool(os.environ.get("GROQ_API_KEY"))
    if PROVIDER == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if PROVIDER == "ollama":
        return _ollama_reachable()
    return False


def _ollama_reachable() -> bool:
    global _ollama_reachable_cache
    if _ollama_reachable_cache is not None:
        return _ollama_reachable_cache
    try:
        import requests
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1.5)
        _ollama_reachable_cache = resp.status_code == 200
    except Exception:
        _ollama_reachable_cache = False
    return _ollama_reachable_cache


def reset_ollama_reachability_cache() -> None:
    """Call this if you've just started Ollama and don't want to wait for
    the next process restart to pick it up (e.g. in a long-running
    uvicorn --reload session)."""
    global _ollama_reachable_cache
    _ollama_reachable_cache = None


def active_model() -> str:
    if PROVIDER == "groq":
        return GROQ_MODEL
    if PROVIDER == "ollama":
        return OLLAMA_MODEL
    return ANTHROPIC_MODEL


def setup_hint() -> str:
    """Provider-specific instructions for the Demo Mode message — what
    the person actually needs to do differs meaningfully between 'add an
    API key' and 'start a local server'."""
    if PROVIDER == "ollama":
        return f"Run `ollama serve` and `ollama pull {OLLAMA_MODEL}`, then ask again"
    return "Add a key to `.env`"


def chat(messages: list[dict], system: str | None = None, max_tokens: int = 800) -> str:
    """messages: [{"role": "user"|"assistant", "content": str}, ...]
    Returns the assistant's reply as plain text. Raises if the active
    provider isn't usable — callers should check api_key_configured()
    first if they want graceful degradation instead of an exception."""
    if PROVIDER == "groq":
        return _chat_groq(messages, system, max_tokens)
    if PROVIDER == "anthropic":
        return _chat_anthropic(messages, system, max_tokens)
    if PROVIDER == "ollama":
        return _chat_ollama(messages, system, max_tokens)
    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER!r} (expected 'groq', 'anthropic', or 'ollama')")


def _chat_groq(messages: list[dict], system: str | None, max_tokens: int) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        messages=full_messages,
    )
    return response.choices[0].message.content or ""


def _chat_anthropic(messages: list[dict], system: str | None, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    kwargs = {"model": ANTHROPIC_MODEL, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    return "".join(block.text for block in response.content if hasattr(block, "text"))


def _chat_ollama(messages: list[dict], system: str | None, max_tokens: int) -> str:
    import requests

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": full_messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        },
        timeout=120,  # local generation on CPU can be slow, especially for longer answers
    )
    response.raise_for_status()
    return response.json()["message"]["content"]
