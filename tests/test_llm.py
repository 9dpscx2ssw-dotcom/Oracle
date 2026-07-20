"""LLM client providers: Anthropic Claude, local Ollama, or disabled."""

from __future__ import annotations


from gungnir.config import Config, Secrets
from gungnir.llm.client import (
    NullLLMClient, ClaudeClient, OllamaClient, build_llm, _extract_json,
)


def test_extract_json_handles_wrapped_replies():
    """Claude/Ollama often wrap JSON in prose or ```json fences — a raw
    json.loads fails with 'Expecting value: line 1 column 1 (char 0)'. The
    extractor recovers the object; unrecoverable input yields {}."""
    assert _extract_json('{"score": 0.4}') == {"score": 0.4}
    assert _extract_json('```json\n{"score": 0.4}\n```') == {"score": 0.4}
    assert _extract_json('```\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('Here it is:\n{"s": -0.2, "c": 0.6}\ndone') == {"s": -0.2, "c": 0.6}
    assert _extract_json('{"a": {"b": 1}, "c": 2}') == {"a": {"b": 1}, "c": 2}
    # Unrecoverable → empty dict (fallback, never raises)
    assert _extract_json('') == {}
    assert _extract_json('I cannot help with that.') == {}


def _config(provider: str = "anthropic", **llm_opts) -> Config:
    """Build a test config with given LLM provider."""
    llm_config = {"provider": provider}
    llm_config.update(llm_opts)
    return Config({"llm": llm_config}, Secrets.from_env())


def test_build_llm_selects_provider():
    """build_llm picks the right provider based on config."""
    # Null provider (disabled)
    c = _config("none")
    assert isinstance(build_llm(c), NullLLMClient)

    # Anthropic (if API key present; otherwise falls back to null)
    c = _config("anthropic")
    provider = build_llm(c)
    assert isinstance(provider, (ClaudeClient, NullLLMClient))

    # Ollama (always returns OllamaClient; can fail gracefully on unavailable server)
    c = _config("ollama")
    provider = build_llm(c)
    assert isinstance(provider, OllamaClient)


def test_ollama_config_passthrough():
    """OllamaClient respects config for URL and model name."""
    c = _config("ollama", ollama_url="http://example.com:11434", ollama_model="neural-chat")
    client = OllamaClient(c)
    assert client.base_url == "http://example.com:11434"
    assert client.model_name == "neural-chat"


def test_ollama_defaults():
    """OllamaClient uses defaults when config is missing."""
    c = Config({}, Secrets.from_env())
    client = OllamaClient(c)
    assert client.base_url == "http://localhost:11434"
    # Small default sized for an 8GB homelab box (TrueNAS etc.).
    assert client.model_name == "llama3.2:3b"


def test_ollama_graceful_failure():
    """OllamaClient returns {} fallback when server is unreachable."""
    c = _config("ollama", ollama_url="http://localhost:9999")
    client = OllamaClient(c)
    # Should not raise; returns empty dict fallback
    result = client.complete_json("test prompt")
    assert result == {}


def test_null_client_always_returns_empty():
    """NullLLMClient is a no-op for tests/disabled mode."""
    client = NullLLMClient()
    assert client.complete_json("anything") == {}
    assert client.complete_json("test", system="system") == {}


def test_ollama_caches_results():
    """OllamaClient caches identical prompts within TTL (bounded LRU)."""
    c = _config("ollama", ollama_url="http://localhost:9999")
    client = OllamaClient(c)
    client._cache.put("test_key", {"cached": True})
    assert client._cache.get("test_key") == {"cached": True}
    assert client._cache.get("missing") is None


def test_llm_cache_is_bounded():
    """The LRU cache evicts oldest entries instead of growing without bound —
    prompts embed per-loop feature values, so keys are nearly always unique."""
    from gungnir.llm.client import _LRUCache
    cache = _LRUCache(ttl=3600, maxsize=8)
    for i in range(50):
        cache.put(f"k{i}", {"i": i})
    assert len(cache._d) == 8
    assert cache.get("k0") is None          # evicted
    assert cache.get("k49") == {"i": 49}    # newest kept


def test_llm_cache_respects_ttl():
    from gungnir.llm.client import _LRUCache
    cache = _LRUCache(ttl=0.0)
    cache.put("k", {"v": 1})
    assert cache.get("k") is None           # already expired at ttl=0
