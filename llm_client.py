from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List

from .json_utils import extract_json_object, strip_think_blocks
from .schema import QwenLLMConfig


class QwenLLMClient:
    """Minimal OpenAI-compatible client implemented with the standard library."""

    def __init__(self, config: QwenLLMConfig):
        self.config = config
        self._available = bool(config.enabled)

    @property
    def available(self) -> bool:
        return self._available

    def _should_disable_thinking(self) -> bool:
        model_name = str(self.config.model or "").lower()
        base_url = str(self.config.base_url or "").lower()
        return "qwen" in model_name or "qwen" in base_url

    def _with_no_think(self, prompt: str) -> str:
        if "/no_think" in prompt or "/think" in prompt:
            return prompt
        prompt = prompt.rstrip()
        return f"{prompt}\n/no_think" if prompt else "/no_think"

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        # Paper Sec. 3.8: the agent backbone is accessed through an external
        # OpenAI-compatible API; no target T2I model internals are required.
        if not self.available:
            return ""
        effective_user_prompt = user_prompt
        if self._should_disable_thinking():
            effective_user_prompt = self._with_no_think(user_prompt)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": effective_user_prompt},
        ]
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            text = data["choices"][0]["message"].get("content") or ""
            if self.config.strip_thinking:
                text = strip_think_blocks(text)
            return text.strip()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError):
            self._available = False
            return ""

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Dict[str, Any]:
        text = self.chat_text(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not text:
            return {}
        return extract_json_object(text)
