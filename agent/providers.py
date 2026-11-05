"""Provider-neutral chat interface for Ollama, OpenAI-compatible APIs, Anthropic, Gemini, NIM and MiniCoder."""
from __future__ import annotations
import json, os, urllib.request, urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

class ProviderError(RuntimeError): pass

@dataclass
class ChatResponse:
    text: str
    provider: str
    model: str
    raw: Dict[str, Any]

class BaseProvider(ABC):
    name = "base"
    def __init__(self, model: str, **config): self.model, self.config = model, config
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> ChatResponse: ...
    def supports(self, feature: str) -> bool: return feature in {"chat"}

class HTTPProvider(BaseProvider):
    def _post_json(self, url: str, payload: Dict[str, Any], headers=None, timeout=60):
        req = urllib.request.Request(url, json.dumps(payload).encode(), headers={"Content-Type":"application/json", **(headers or {})}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())
        except urllib.error.HTTPError as e: raise ProviderError(f"{self.name} HTTP {e.code}: {e.read().decode(errors='replace')[:1000]}") from e
        except OSError as e: raise ProviderError(f"{self.name} request failed: {e}") from e

class OpenAICompatibleProvider(HTTPProvider):
    name = "openai_compatible"
    def chat(self, messages, **kwargs):
        base = self.config.get("base_url", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")).rstrip("/")
        key = self.config.get("api_key", os.getenv("OPENAI_API_KEY"))
        if not key: raise ProviderError("OpenAI-compatible provider requires api_key or OPENAI_API_KEY")
        raw = self._post_json(base + "/chat/completions", {"model":self.model,"messages":messages,**kwargs}, {"Authorization":"Bearer "+key})
        return ChatResponse(raw.get("choices", [{}])[0].get("message", {}).get("content", ""), self.name, self.model, raw)

class OllamaProvider(HTTPProvider):
    name = "ollama"
    def chat(self, messages, **kwargs):
        url = self.config.get("base_url", "http://127.0.0.1:11434") + "/api/chat"
        raw = self._post_json(url, {"model":self.model,"messages":messages,"stream":False,**kwargs})
        return ChatResponse(raw.get("message", {}).get("content", ""), self.name, self.model, raw)

class AnthropicProvider(HTTPProvider):
    name = "anthropic"
    def chat(self, messages, **kwargs):
        key = self.config.get("api_key", os.getenv("ANTHROPIC_API_KEY"))
        if not key: raise ProviderError("Anthropic provider requires api_key or ANTHROPIC_API_KEY")
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
        body = {"model":self.model,"max_tokens":kwargs.pop("max_tokens",1024),"messages":[m for m in messages if m.get("role") != "system"],**kwargs}
        if system: body["system"] = system
        raw = self._post_json(self.config.get("base_url", "https://api.anthropic.com") + "/v1/messages", body, {"x-api-key":key,"anthropic-version":"2023-06-01"})
        return ChatResponse("".join(x.get("text","") for x in raw.get("content",[])), self.name, self.model, raw)

class GeminiProvider(HTTPProvider):
    name = "gemini"
    def chat(self, messages, **kwargs):
        key = self.config.get("api_key", os.getenv("GOOGLE_API_KEY"))
        if not key: raise ProviderError("Gemini provider requires api_key or GOOGLE_API_KEY")
        contents = [{"role":"model" if m["role"]=="assistant" else "user", "parts":[{"text":m["content"]}]} for m in messages if m["role"] != "system"]
        url = self.config.get("base_url", "https://generativelanguage.googleapis.com/v1beta/models") + f"/{self.model}:generateContent?key={key}"
        raw = self._post_json(url, {"contents":contents,**kwargs})
        text = raw.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","")
        return ChatResponse(text, self.name, self.model, raw)

class NIMProvider(OpenAICompatibleProvider):
    name = "nvidia_nim"
    def chat(self, messages, **kwargs):
        self.config.setdefault("base_url", "https://integrate.api.nvidia.com/v1")
        self.config.setdefault("api_key", os.getenv("NVIDIA_API_KEY"))
        return super().chat(messages, **kwargs)

class MiniCoderProvider(BaseProvider):
    name = "minicoder"
    def __init__(self, model="local", model_runner=None, **config): super().__init__(model, **config); self.model_runner = model_runner
    def chat(self, messages, **kwargs):
        if self.model_runner is None: raise ProviderError("MiniCoder provider requires a model_runner callable; no pretrained intelligence is assumed")
        text = self.model_runner(messages, **kwargs)
        return ChatResponse(str(text), self.name, self.model, {"local":True})

PROVIDERS = {"ollama":OllamaProvider,"openai":OpenAICompatibleProvider,"openai_compatible":OpenAICompatibleProvider,"anthropic":AnthropicProvider,"gemini":GeminiProvider,"nvidia_nim":NIMProvider,"minicoder":MiniCoderProvider}
def create_provider(name: str, model: str, **config) -> BaseProvider:
    try: return PROVIDERS[name.lower()](model, **config)
    except KeyError: raise ValueError(f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}")
