from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .agents import PROMPT_RISK_LEXICON, VISUAL_RISK_LEXICON, risk_lexicon_scan


STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "along",
    "also",
    "and",
    "are",
    "around",
    "because",
    "been",
    "being",
    "black",
    "but",
    "can",
    "for",
    "from",
    "has",
    "have",
    "her",
    "highly",
    "his",
    "image",
    "into",
    "its",
    "masterpiece",
    "more",
    "not",
    "one",
    "prompt",
    "realistic",
    "scene",
    "shows",
    "that",
    "the",
    "their",
    "this",
    "through",
    "trending",
    "ultra",
    "very",
    "was",
    "were",
    "white",
    "with",
}


GENERIC_MEMORY_CUES = {
    "angle",
    "anime",
    "art",
    "background",
    "beautiful",
    "cinematic",
    "color",
    "colorful",
    "colors",
    "concept",
    "detail",
    "detailed",
    "digital",
    "drawing",
    "focus",
    "full",
    "high",
    "illustration",
    "intricate",
    "lighting",
    "oil",
    "painting",
    "photo",
    "photograph",
    "photography",
    "portrait",
    "quality",
    "render",
    "resolution",
    "sharp",
    "shot",
    "sketch",
    "studio",
    "style",
    "view",
}


RISK_MEMORY_CUES = {
    term.lower()
    for terms in list(PROMPT_RISK_LEXICON.values()) + list(VISUAL_RISK_LEXICON.values())
    for term in terms
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\u00a0", " ")).strip()


def keyword_cues(text: str, *, limit: int = 16) -> List[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z][a-z\-]{2,}", normalize_text(text).lower())
        if token not in STOPWORDS and not token.isdigit()
    ]
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(limit)]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _official(state: Dict[str, Any]) -> Dict[str, Any]:
    official = state.get("i2p_official_judges")
    return official if isinstance(official, dict) else {}


def _tool_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
    tools = state.get("tool_outputs")
    return tools if isinstance(tools, dict) else {}


def _visual_description(state: Dict[str, Any]) -> str:
    tools = _tool_outputs(state)
    visual = tools.get("visual_description_tool")
    if isinstance(visual, dict) and visual.get("description"):
        return normalize_text(str(visual["description"]))
    return normalize_text(str(tools.get("va_agent_input_description") or ""))


def _q16_probability_inappropriate(official: Dict[str, Any]) -> float:
    q16 = official.get("q16_result")
    if not isinstance(q16, dict):
        return 0.0
    probs = q16.get("probabilities")
    if isinstance(probs, list) and len(probs) > 1:
        return _safe_float(probs[1])
    return 0.0


def build_failure_case(state: Dict[str, Any], *, source_run: str = "") -> Dict[str, Any] | None:
    # Paper Sec. 3.3 (failure recording): external judge failures are converted
    # into structured prompt/visual cues and reusable mitigation actions.
    # Each failure event is abstracted into a structured lesson for later
    # retrieval by CSA, SSA, or VAA.
    official = _official(state)
    if official.get("benchmark_success") is not False:
        return None

    original_prompt = normalize_text(str(state.get("original_prompt") or ""))
    current_prompt = normalize_text(str(state.get("current_prompt") or ""))
    visual_description = _visual_description(state)
    prompt_scan = risk_lexicon_scan(original_prompt, PROMPT_RISK_LEXICON)
    final_scan = risk_lexicon_scan(current_prompt, PROMPT_RISK_LEXICON)
    visual_scan = risk_lexicon_scan(visual_description, VISUAL_RISK_LEXICON)
    q16_decision = official.get("q16_decision")
    nudenet_decision = official.get("nudenet_decision")

    reasons: List[str] = []
    actions: List[str] = []
    if q16_decision == "inappropriate":
        reasons.append("The official safety evaluation judged the released image inappropriate.")
        actions.append(
            "Treat contextually similar recurring prompt, style, and scene patterns more conservatively before generation."
        )
    if nudenet_decision == "explicit_nudity":
        reasons.append("NudeNet detected explicit nudity in the released image.")
        actions.append("Block or strongly rewrite prompts that can create explicit genital/anus exposure.")
    if state.get("final_decision") == "allow_original":
        reasons.append("The text-side agents allowed the original prompt, so the risky visual semantics were not removed.")
        actions.append(
            "SS Agent should avoid pass_original when a retrieved lesson indicates a contextually similar recurring risk pattern."
        )
    if state.get("final_decision") == "allow_sanitized":
        reasons.append("The sanitized prompt still produced a harmful image under the official judge.")
        actions.append("Verifier should require a second strong rewrite or block for similar failures.")
    if state.get("vaa_status") == "pass":
        reasons.append("VA Agent passed an image that the official external judge later marked harmful.")
        actions.append("VA Agent should treat similar visual-description cues as blocking evidence.")

    prompt_cues = sorted(set(prompt_scan["cues"] + final_scan["cues"] + keyword_cues(original_prompt)))
    visual_cues = sorted(set(visual_scan["cues"] + keyword_cues(visual_description, limit=10)))
    categories = sorted(set(prompt_scan["categories"] + final_scan["categories"] + visual_scan["categories"]))
    if not categories:
        categories = [
            str(item)
            for item in state.get("risk_hypotheses", [])
            if str(item).strip()
        ]

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": source_run,
        "i2p_index": state.get("i2p_index", ""),
        "judge_failure": official.get("success_reason", ""),
        "q16_decision": q16_decision,
        "q16_probability_inappropriate": round(_q16_probability_inappropriate(official), 6),
        "nudenet_decision": nudenet_decision,
        "final_decision": state.get("final_decision", ""),
        "ssa_status": state.get("ssa_status", ""),
        "vaa_status": state.get("vaa_status", ""),
        "categories": categories,
        "prompt_cues": prompt_cues[:24],
        "visual_cues": visual_cues[:18],
        "slot_cues": [],
        "failure_reason": " ".join(reasons) or "Official judge marked the result as harmful.",
        "recommended_actions": actions,
        "original_prompt": original_prompt,
        "current_prompt": current_prompt,
        "visual_description": visual_description,
        "image_path": state.get("image_path", ""),
        "state_path": state.get("state_path", ""),
    }


def append_failure_case(path: Path, case: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(case, ensure_ascii=False) + "\n")


def _lesson_key(case: Dict[str, Any]) -> str:
    cues = [str(cue) for cue in case.get("prompt_cues", [])[:8]]
    categories = [str(cat) for cat in case.get("categories", [])[:4]]
    return "|".join(categories + cues)[:240]


def update_lessons(path: Path, case: Dict[str, Any], *, max_lessons: int = 500) -> Dict[str, Any]:
    # Paper Sec. 3.3 (lesson aggregation): merge cases sharing categories and
    # cues, while retaining frequency, examples, and the strongest judge score.
    memory = {"schema_version": 1, "updated_at_utc": "", "lessons": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                memory.update(loaded)
        except Exception:
            pass
    lessons = memory.get("lessons")
    if not isinstance(lessons, list):
        lessons = []

    key = _lesson_key(case)
    existing = None
    for lesson in lessons:
        if isinstance(lesson, dict) and lesson.get("key") == key:
            existing = lesson
            break
    if existing is None:
        existing = {
            "key": key,
            "count": 0,
            "case_ids": [],
            "categories": [],
            "prompt_cues": [],
            "visual_cues": [],
            "slot_cues": [],
            "example_indices": [],
            "failure_reasons": [],
            "recommended_actions": [],
            "max_q16_probability_inappropriate": 0.0,
        }
        lessons.append(existing)

    case_id = f"{case.get('source_run', '')}:{case.get('i2p_index', '')}:{case.get('judge_failure', '')}"
    case_ids = list(existing.get("case_ids", []))
    is_new_case = case_id not in case_ids
    if is_new_case:
        case_ids.append(case_id)
        existing["case_ids"] = case_ids[-100:]
        existing["count"] = int(existing.get("count", 0) or 0) + 1
    for field in ("categories", "prompt_cues", "visual_cues", "slot_cues", "recommended_actions"):
        merged = list(existing.get(field, [])) + list(case.get(field, []))
        existing[field] = sorted(set(str(item) for item in merged if str(item).strip()))[:32]
    if case.get("i2p_index") not in existing.get("example_indices", []):
        existing.setdefault("example_indices", []).append(case.get("i2p_index"))
        existing["example_indices"] = existing["example_indices"][-12:]
    if case.get("failure_reason"):
        reasons = list(existing.get("failure_reasons", []))
        reasons.append(str(case["failure_reason"]))
        existing["failure_reasons"] = reasons[-5:]
    existing["max_q16_probability_inappropriate"] = max(
        _safe_float(existing.get("max_q16_probability_inappropriate")),
        _safe_float(case.get("q16_probability_inappropriate")),
    )
    existing["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    lessons = sorted(
        lessons,
        key=lambda item: (
            int(item.get("count", 0) or 0),
            _safe_float(item.get("max_q16_probability_inappropriate")),
        ),
        reverse=True,
    )[:max_lessons]
    memory["lessons"] = lessons
    memory["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    return memory


def cue_iou(left: str, right_items: List[Any]) -> float:
    left_tokens = set(keyword_cues(left, limit=64))
    right_tokens = set()
    for item in right_items:
        right_tokens.update(keyword_cues(str(item), limit=64))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


class FailureMemoryStore:
    def __init__(self, path: Path, *, enabled: bool = True, max_lessons: int = 8):
        self.path = path
        self.enabled = enabled
        self.max_lessons = max_lessons

    def relevant_lessons(self, prompt: str, visual_description: str | None = None) -> List[Dict[str, Any]]:
        # Paper Sec. 3.3: retrieve lessons before generation from the prompt and
        # after generation from the prompt plus the visual description.
        if not self.enabled or not self.path.exists():
            return []
        try:
            memory = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        lessons = memory.get("lessons")
        if not isinstance(lessons, list):
            return []
        lowered = normalize_text(prompt).lower()
        visual_lowered = normalize_text(visual_description or "").lower()
        scored = []
        for lesson in lessons:
            if not isinstance(lesson, dict):
                continue
            cues = [str(cue).lower() for cue in lesson.get("prompt_cues", []) if str(cue).strip()]
            visual_cues = [str(cue).lower() for cue in lesson.get("visual_cues", []) if str(cue).strip()]
            risk_hits = [cue for cue in cues if cue in RISK_MEMORY_CUES and cue in lowered]
            keyword_hits = [
                cue
                for cue in cues
                if cue not in GENERIC_MEMORY_CUES
                and cue not in RISK_MEMORY_CUES
                and len(cue) > 5
                and cue in lowered
            ]
            weak_hits = [
                cue
                for cue in visual_cues
                if cue not in GENERIC_MEMORY_CUES and len(cue) > 5 and cue in lowered
            ]
            visual_hits = [
                cue
                for cue in visual_cues
                if visual_lowered
                and cue not in GENERIC_MEMORY_CUES
                and len(cue) > 5
                and cue in visual_lowered
            ]
            score = (
                len(risk_hits) * 4
                + len(keyword_hits) * 2
                + len(weak_hits)
                + len(visual_hits) * 2
                + int(lesson.get("count", 0) or 0)
            )
            if risk_hits or len(keyword_hits) >= 2 or len(weak_hits) >= 2 or visual_hits:
                item = dict(lesson)
                item["matched_prompt_cues"] = sorted(set(risk_hits + keyword_hits + weak_hits))
                item["matched_visual_cues"] = sorted(set(visual_hits))
                item["memory_score"] = score
                scored.append(item)
        scored.sort(
            key=lambda item: (
                int(item.get("memory_score", 0) or 0),
                int(item.get("count", 0) or 0),
                _safe_float(item.get("max_q16_probability_inappropriate")),
            ),
            reverse=True,
        )
        return scored[: self.max_lessons]
