"""
LLM provider integrations — uniform interface for OpenAI, Gemini, Claude, and Groq.
"""
from __future__ import annotations
import time


PROVIDER_MODELS: dict[str, list[str]] = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "Gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    "Claude": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"],
    "Groq (Free)": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
}

# Messages format: list of {"role": "system"|"user"|"assistant", "content": str}
Messages = list[dict[str, str]]


def call_llm(
    provider: str,
    model: str,
    api_key: str,
    messages: Messages,
    json_mode: bool = False,
    temperature: float = 0.7,
) -> str:
    """
    Dispatch to the appropriate LLM provider and return the raw text response.

    Args:
        provider: One of the keys in PROVIDER_MODELS.
        model: Model identifier string.
        api_key: Provider API key.
        messages: Conversation history in OpenAI-style message format.
        json_mode: Request structured JSON output where natively supported.
        temperature: Sampling temperature. Higher (≈0.9) for diverse question
            generation; 0.0 for deterministic answer verification.

    Returns:
        Raw string response from the model.

    Raises:
        ValueError: On authentication, quota, or response errors.
    """
    try:
        if provider == "OpenAI":
            return _call_openai(model, api_key, messages, json_mode, temperature)
        elif provider == "Gemini":
            return _call_gemini(model, api_key, messages, temperature)
        elif provider == "Claude":
            return _call_claude(model, api_key, messages, temperature)
        elif provider in ("Groq", "Groq (Free)"):
            return _call_groq(model, api_key, messages, json_mode, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider!r}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"[{provider}] API error: {exc}") from exc


# ─── OpenAI ──────────────────────────────────────────────────────────────────

def _call_openai(
    model: str, api_key: str, messages: Messages, json_mode: bool, temperature: float
) -> str:
    from openai import OpenAI, AuthenticationError, RateLimitError

    client = OpenAI(api_key=api_key)
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
    except AuthenticationError:
        raise ValueError("OpenAI: invalid API key. Check your key at platform.openai.com.")
    except RateLimitError:
        raise ValueError("OpenAI: rate limit or quota exceeded.")

    return resp.choices[0].message.content or ""


# ─── Claude ──────────────────────────────────────────────────────────────────

def _call_claude(model: str, api_key: str, messages: Messages, temperature: float) -> str:
    from anthropic import Anthropic, AuthenticationError

    client = Anthropic(api_key=api_key)

    system_content: str | None = None
    chat_messages: Messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})

    kwargs: dict = {
        "model": model,
        "max_tokens": 8192,
        "temperature": temperature,
        "messages": chat_messages,
    }
    if system_content:
        kwargs["system"] = system_content

    try:
        resp = client.messages.create(**kwargs)
    except AuthenticationError:
        raise ValueError("Claude: invalid API key. Check your key at console.anthropic.com.")

    return resp.content[0].text


# ─── Gemini ──────────────────────────────────────────────────────────────────

def _call_gemini(model: str, api_key: str, messages: Messages, temperature: float) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    system_parts: list[str] = []
    history: list[dict] = []

    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        elif msg["role"] == "user":
            history.append({"role": "user", "parts": [msg["content"]]})
        elif msg["role"] == "assistant":
            history.append({"role": "model", "parts": [msg["content"]]})

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    gen_config = {"temperature": temperature}

    try:
        model_obj = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction,
            generation_config=gen_config,
        )
    except Exception:
        # Older SDK versions may not accept system_instruction / generation_config
        model_obj = genai.GenerativeModel(model_name=model)

    try:
        if len(history) > 1:
            chat = model_obj.start_chat(history=history[:-1])
            resp = chat.send_message(history[-1]["parts"][0])
        elif history:
            resp = model_obj.generate_content(history[0]["parts"][0])
        else:
            raise ValueError("Gemini: no user message found in messages list.")

        return resp.text

    except Exception as exc:
        err = str(exc).lower()
        if "api_key" in err or "invalid" in err or "unauthorized" in err:
            raise ValueError("Gemini: invalid API key. Check your key at aistudio.google.com.")
        raise ValueError(f"Gemini: {exc}")


# ─── Groq ────────────────────────────────────────────────────────────────────

def _call_groq(
    model: str, api_key: str, messages: Messages, json_mode: bool, temperature: float
) -> str:
    from groq import Groq, AuthenticationError, RateLimitError

    client = Groq(api_key=api_key)
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
    except AuthenticationError:
        raise ValueError("Groq: invalid API key. Check your key at console.groq.com.")
    except RateLimitError:
        raise ValueError("Groq: free-tier rate limit hit. Wait a moment and try again.")

    return resp.choices[0].message.content or ""


# ─── Validation helper ────────────────────────────────────────────────────────

def validate_api_key(provider: str, model: str, api_key: str) -> tuple[bool, str]:
    """Make a minimal test call to verify the API key works."""
    try:
        call_llm(
            provider=provider,
            model=model,
            api_key=api_key,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
        )
        return True, "API key validated successfully."
    except ValueError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Unexpected error during validation: {exc}"
