"""Provider-agnostic LLM client with an Anthropic Claude implementation.

Design goals for LLM in a trading loop:
  • Rate-limit aware  — Anthropic tier caps requests/minute. A token-bucket guards it.
  • Cached            — identical prompts within a TTL reuse the last answer.
  • Fail-safe         — any error returns a structured fallback, never raises into
                        the trading loop. The agent must keep trading if the LLM
                        is down or throttled.
  • JSON-first        — callers ask for JSON and get a parsed dict.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

from ..config import Config

log = logging.getLogger(__name__)

# Prompts embed per-loop feature values, so most cache keys are unique — an
# unbounded dict was a slow memory leak that never actually hit. LRU-bound it.
_CACHE_MAX = 512


class _LRUCache:
    """TTL'd LRU over (monotonic_ts, value) entries, bounded to _CACHE_MAX."""

    def __init__(self, ttl: float, maxsize: int = _CACHE_MAX):
        self.ttl = float(ttl)
        self.maxsize = maxsize
        self._d: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    def get(self, key: str) -> dict | None:
        hit = self._d.get(key)
        if hit is None or (time.monotonic() - hit[0]) >= self.ttl:
            return None
        self._d.move_to_end(key)
        return hit[1]

    def put(self, key: str, value: dict) -> None:
        self._d[key] = (time.monotonic(), value)
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)


def _extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object from an LLM reply.

    Models frequently wrap JSON in prose ("Here is the analysis:") or ```json
    code fences despite instructions, so a bare json.loads fails with
    'Expecting value: line 1 column 1 (char 0)'. Fall back to the outermost
    {...} span, which handles both fenced and prose-wrapped replies.
    """
    if not text:
        return {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return {}


class LLMClient(ABC):
    @abstractmethod
    def complete_json(self, prompt: str, *, system: str = "",
                      max_tokens: int = 1024) -> dict:
        """Return a parsed JSON object, or {} on failure. ``max_tokens`` caps the
        reply — pass a small budget for small-schema tasks (sentiment is 3 fields;
        4096 tokens of headroom is pure latency)."""


class _RateLimiter:
    def __init__(self, max_per_minute: int):
        self.min_interval = 60.0 / max(1, max_per_minute)
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


class ClaudeClient(LLMClient):
    def __init__(self, config: Config):
        self.model_name = config.secrets.anthropic_model
        self.api_key = config.secrets.anthropic_api_key
        self.limiter = _RateLimiter(config.get("llm", "max_calls_per_minute", default=12))
        self.cache_ttl = config.get("llm", "cache_ttl_seconds", default=900)
        self.cooldown_seconds = config.get("llm", "quota_cooldown_seconds", default=300)
        self._cooldown_until = 0.0
        self._cache = _LRUCache(self.cache_ttl)
        self._client = None
        if self.api_key:
            self._init_client()
        else:
            log.warning("ANTHROPIC_API_KEY not set; LLM calls will return fallbacks.")

    def _init_client(self) -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=self.api_key)

    def complete_json(self, prompt: str, *, system: str = "",
                      max_tokens: int = 1024) -> dict:
        key = hashlib.sha256(f"{system}\n{prompt}".encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self._client is None:
            return {}

        # If we recently hit a quota/rate limit, skip the call entirely until the
        # cooldown elapses — no point hammering a dead quota every loop.
        if time.monotonic() < self._cooldown_until:
            return {}

        self.limiter.wait()
        try:
            resp = self._client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system=system or None,
                messages=[{"role": "user", "content": prompt}],
            )
            # Claude returns content as a list; extract text from the first block.
            content = resp.content[0].text if resp.content else "{}"
            # Robust parse: Claude often adds prose or ```json fences around the
            # object, which raw json.loads rejects ("Expecting value ... char 0").
            data = _extract_json(content)
        except Exception as exc:  # noqa: BLE001 — LLM must never break the loop
            text = str(exc)
            low = text.lower()
            if ("429" in text or "rate_limit" in low or "credit balance" in low
                    or "billing" in low or "authentication" in low or "401" in text):
                # Quota, billing, or auth failure: every subsequent call would fail
                # the same way, so back off instead of paying the rate-limiter wait
                # (and spamming logs) on each of them.
                self._cooldown_until = time.monotonic() + self.cooldown_seconds
                log.warning(
                    "LLM unavailable (quota/billing/auth); pausing LLM calls for %ds "
                    "(check Anthropic billing/quota or ANTHROPIC_API_KEY). Error: %s",
                    self.cooldown_seconds, text.splitlines()[0][:200],
                )
            else:
                # Compact: provider errors can be multi-KB JSON blobs.
                first = text.splitlines()[0][:200] if text else type(exc).__name__
                log.warning("LLM call failed (%s); returning fallback.", first)
            return {}

        self._cache.put(key, data)
        return data


class OllamaClient(LLMClient):
    """Local LLM via Ollama: zero cost, fully private, offline-capable."""

    def __init__(self, config: Config):
        self.base_url = config.get("llm", "ollama_url", default="http://localhost:11434")
        # Small default: fits an 8GB homelab box (e.g. TrueNAS, where ZFS already
        # claims much of RAM) and is plenty for the simple JSON sentiment/
        # prediction tasks. Bump to a 7B (mistral, llama3.1) only with the RAM.
        self.model_name = config.get("llm", "ollama_model", default="llama3.2:3b")
        self.cache_ttl = config.get("llm", "cache_ttl_seconds", default=900)
        self._cache = _LRUCache(self.cache_ttl)
        self._check_available()

    def _check_available(self) -> None:
        """Verify Ollama is running and the configured model is available."""
        try:
            import httpx
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            r.raise_for_status()
            installed = [m.get("name", "") for m in r.json().get("models", [])]
            # Match exactly (e.g. "llama3.2:3b") OR by base name so a configured
            # "llama3.2" still recognizes an installed "llama3.2:latest". Don't
            # strip the tag off only one side — that misfires a false warning
            # whenever the configured name carries a tag.
            base = self.model_name.split(":")[0]
            found = any(
                name == self.model_name or name.split(":")[0] == base
                for name in installed
            )
            if not found:
                log.warning(
                    "Model '%s' not available in Ollama. Pull it with: "
                    "ollama pull %s", self.model_name, self.model_name
                )
        except Exception as e:
            log.warning(
                "Ollama unavailable at %s: %s — run 'ollama serve' first.",
                self.base_url, e
            )

    def complete_json(self, prompt: str, *, system: str = "",
                      max_tokens: int = 1024) -> dict:
        key = hashlib.sha256(f"{system}\n{prompt}".encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        try:
            import httpx
            full_prompt = f"{system}\n{prompt}" if system else prompt
            r = httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": full_prompt,
                      "stream": False, "format": "json",
                      "options": {"num_predict": max_tokens}},
                timeout=60.0,
            )
            r.raise_for_status()
            text = r.json().get("response", "{}")
            data = _extract_json(text)
        except Exception as e:  # noqa: BLE001
            log.warning("Ollama call failed (%s); returning fallback.", str(e)[:200])
            return {}

        self._cache.put(key, data)
        return data


class CodexClient(LLMClient):
    """LLM via the OpenAI Codex CLI — rides a ChatGPT-plan subscription.

    Runs ``codex exec`` as a subprocess for each (uncached) call, so the only
    setup is ``codex login`` once on the host (OAuth; no API key in .env). In
    Docker, mount the host's ``~/.codex`` into the container so the token is
    visible. Suits Gungnir's LLM profile exactly: advisory-only, cached,
    background-threaded, never on the trading loop's critical path — multi-
    second CLI latency is irrelevant here.

    Notes:
      • ``max_tokens`` is advisory only (the CLI offers no direct cap); the
        prompts already ask for small JSON objects.
      • Plan usage limits behave like quota errors — repeated failures trip
        the same cooldown as the Anthropic client so a dead quota isn't
        hammered every loop.
    """

    def __init__(self, config: Config):
        self.cmd = str(config.get("llm", "codex_cmd", default="codex"))
        self.model_name = str(config.get("llm", "codex_model", default="") or "")
        self.timeout = float(config.get("llm", "codex_timeout_seconds", default=120) or 120)
        self.cache_ttl = config.get("llm", "cache_ttl_seconds", default=900)
        self.cooldown_seconds = config.get("llm", "quota_cooldown_seconds", default=300)
        self._cooldown_until = 0.0
        self._failures = 0
        self._cache = _LRUCache(self.cache_ttl)

    def complete_json(self, prompt: str, *, system: str = "",
                      max_tokens: int = 1024) -> dict:
        key = hashlib.sha256(f"{system}\n{prompt}".encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if time.monotonic() < self._cooldown_until:
            return {}

        import subprocess
        import tempfile
        from pathlib import Path
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        out_file = Path(tempfile.mktemp(prefix="codex_", suffix=".txt"))
        argv = [self.cmd, "exec", "--skip-git-repo-check",
                "--output-last-message", str(out_file)]
        if self.model_name:
            argv += ["-m", self.model_name]
        argv.append(full_prompt)
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=self.timeout)
            text = out_file.read_text() if out_file.exists() else proc.stdout
            data = _extract_json(text)
            if proc.returncode != 0 and not data:
                raise RuntimeError(
                    (proc.stderr or proc.stdout or "codex exec failed")
                    .strip().splitlines()[-1][:200])
            self._failures = 0
        except FileNotFoundError:
            self._cooldown_until = time.monotonic() + self.cooldown_seconds
            log.warning("Codex CLI '%s' not found — install it and run "
                        "'codex login' (pausing LLM calls %ds).",
                        self.cmd, self.cooldown_seconds)
            return {}
        except Exception as e:  # noqa: BLE001 — LLM must never break the loop
            self._failures += 1
            if self._failures >= 3:
                # Repeated failures look like an expired login or exhausted
                # plan quota — back off instead of paying a 2-min subprocess
                # timeout on every call.
                self._cooldown_until = time.monotonic() + self.cooldown_seconds
                self._failures = 0
                log.warning("Codex CLI failing repeatedly (%s); pausing LLM "
                            "calls for %ds (check 'codex login' / plan quota).",
                            str(e)[:200], self.cooldown_seconds)
            else:
                log.warning("Codex CLI call failed (%s); returning fallback.",
                            str(e)[:200])
            return {}
        finally:
            try:
                out_file.unlink(missing_ok=True)
            except OSError:
                pass

        self._cache.put(key, data)
        return data


class NullLLMClient(LLMClient):
    """No-op client for dry-run/tests; always returns an empty/neutral result."""

    def complete_json(self, prompt: str, *, system: str = "",
                      max_tokens: int = 1024) -> dict:
        return {}


def build_llm(config: Config) -> LLMClient:
    provider = config.get("llm", "provider", default="anthropic")
    if provider == "ollama":
        return OllamaClient(config)
    if provider == "codex":
        return CodexClient(config)
    if provider == "anthropic" and config.secrets.anthropic_api_key:
        return ClaudeClient(config)
    return NullLLMClient()
