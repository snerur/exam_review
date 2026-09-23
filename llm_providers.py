"""
LLM provider integrations — uniform interface for OpenAI, Gemini, Claude, and Groq.

Cost-minimisation techniques applied:
  - Claude: ephemeral prompt caching (cache_control) on system messages.  When the
    same system prompt re-appears across batches or the subsequent verification pass
    (within the 5-minute TTL) the cached tokens are billed at ~10 % of the normal
    rate, yielding roughly 90 % savings on repeated system-prompt tokens.
  - All providers: conservative max_tokens caps prevent runaway charges.
  - Gemini: uses the current google-genai SDK (v1.x) — the old google-generativeai
    package is deprecated and no longer receives API compatibility updates.
  - Groq: free-tier rate-limit delays are handled upstream in exam_generator.py.
"""
from __future__ import annotations


PROVIDER_MODELS: dict[str, list[str]] = {
    "OpenAI": ["gpt-4.1-mini", "gpt-4.1", "gpt-5-mini", "gpt-5.2", "gpt-4o-mini", "gpt-4o"],
    "Gemini": [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.1-pro-preview",
    ],
    "Claude": [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-5",
        "claude-opus-5-5",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
    ],
    "Groq (Free)": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
}

def supports_temperature(provider: str, model: str) -> bool:
    """Only send sampling overrides to known compatible model families.

    Reasoning models use their defaults. Gemini 3 also recommends retaining
    the default temperature rather than lowering it for verification.
    """
    if provider == "OpenAI":
        return model.startswith(("gpt-4o", "gpt-4.1", "gpt-3.5-turbo"))
    if provider in ("Claude", "Anthropic"):
        return model.startswith(("claude-haiku-4-5", "claude-sonnet-4-6"))
    if provider == "Gemini":
        return model.removeprefix("models/").startswith("gemini-2.5-")
    return provider in ("Groq", "Groq (Free)")


def _require_text(text: str | None, provider: str) -> str:
    if not text or not text.strip():
        raise ValueError(
            f"{provider}: the model returned no text (it may have been blocked "
            "or exhausted its output budget). Try again or choose another model."
        )
    return text


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
        temperature: Sampling preference; omitted when unsupported or discouraged.

    Returns:
        Raw string response from the model.

    Raises:
        ValueError: On authentication, quota, or response errors.
    """
    try:
        if provider == "OpenAI":
            return _call_openai(model, api_key, messages, json_mode, temperature)
        elif provider == "Gemini":
            return _call_gemini(model, api_key, messages, temperature, json_mode)
        elif provider in ("Claude", "Anthropic"):
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
        "max_completion_tokens": 16384,
    }
    if supports_temperature("OpenAI", model):
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
    except AuthenticationError:
        raise ValueError("OpenAI: invalid API key. Check your key at platform.openai.com.")
    except RateLimitError:
        raise ValueError("OpenAI: rate limit or quota exceeded.")

    return _require_text(resp.choices[0].message.content, "OpenAI")


# ─── Claude (Anthropic) ───────────────────────────────────────────────────────

def _call_claude(model: str, api_key: str, messages: Messages, temperature: float) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    system_content: str | None = None
    chat_messages: list[dict] = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})

    kwargs: dict = {
        "model": model,
        "max_tokens": 8192,
        "messages": chat_messages,
    }

    if supports_temperature("Claude", model):
        kwargs["temperature"] = temperature

    # Ephemeral prompt caching: the system block is stored server-side for 5 minutes.
    # Back-to-back batches and the verification pass reuse the cache, saving ~90 %
    # on those tokens.  No extra latency — the first call populates the cache.
    if system_content:
        kwargs["system"] = [
            {
                "type": "text",
                "text": system_content,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    try:
        resp = client.messages.create(**kwargs)
    except anthropic.AuthenticationError:
        raise ValueError("Claude: invalid API key. Check your key at console.anthropic.com.")
    except anthropic.RateLimitError:
        raise ValueError("Claude: rate limit exceeded. Please wait and try again.")

    return _require_text("".join(block.text for block in resp.content if block.type == "text"), "Claude")


# ─── Gemini (google-genai SDK v1.x) ──────────────────────────────────────────

def _call_gemini(
    model: str, api_key: str, messages: Messages, temperature: float, json_mode: bool = False
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, vertexai=False)

    system_parts: list[str] = []
    contents: list[types.Content] = []

    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        elif msg["role"] == "user":
            contents.append(
                types.Content(role="user", parts=[types.Part(text=msg["content"])])
            )
        elif msg["role"] == "assistant":
            # Gemini uses "model" role for assistant turns
            contents.append(
                types.Content(role="model", parts=[types.Part(text=msg["content"])])
            )

    system_instruction = "\n\n".join(system_parts) if system_parts else None

    config_kwargs: dict = {"max_output_tokens": 16384}
    if supports_temperature("Gemini", model):
        config_kwargs["temperature"] = temperature
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    config = types.GenerateContentConfig(**config_kwargs)

    try:
        resp = client.models.generate_content(
            model=model.removeprefix("models/"),
            contents=contents,
            config=config,
        )
        return _require_text(resp.text, "Gemini")
    except ValueError:
        raise
    except Exception as exc:
        code = getattr(exc, "code", None)
        err = str(exc).lower()
        if "api_key_invalid" in err or "api key not valid" in err or code == 401:
            raise ValueError("Gemini: invalid API key. Check your key at aistudio.google.com.") from exc
        if code == 404:
            raise ValueError(f"Gemini: model {model!r} is unavailable. Select another model.") from exc
        if code == 403:
            raise ValueError("Gemini: permission denied. Check model access and API project permissions.") from exc
        if code == 429:
            raise ValueError("Gemini: rate limit or quota exceeded.") from exc
        raise ValueError(f"Gemini: {exc}") from exc
    finally:
        client.close()



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
        return True, f"Connection to {provider} / {model} validated successfully."
    except ValueError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Unexpected error during validation: {exc}"
