from __future__ import annotations

import json
import re
from typing import Any, Dict


def strip_think_blocks(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    match = re.search(r"<think>", cleaned, flags=re.IGNORECASE)
    if not match:
        return cleaned.strip()

    start = match.start()
    json_start = cleaned.find("{", start)
    if json_start != -1:
        cleaned = cleaned[:start] + cleaned[json_start:]
    else:
        cleaned = cleaned[:start]
    return cleaned.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_think_blocks(text)
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        payload = json.loads(cleaned[start : end + 1])
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
