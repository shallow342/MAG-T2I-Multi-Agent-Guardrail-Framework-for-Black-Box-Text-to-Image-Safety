from __future__ import annotations

import math
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .llm_client import QwenLLMClient
from .schema import AgentEvidence, AgentVersionConfig, DefenseState


RISK_CATEGORIES = {
    "sexual",
    "violence",
    "self-harm",
    "hate",
    "drugs",
    "minors",
    "exploitation",
    "jailbreak",
}

EXPLICIT_TERMS_PATH = Path(__file__).with_name("explict.json")

SUGGESTIVE_TERMS_PATH = Path(__file__).with_name("suggestive.json")

# Paper Sec. 3.5(a): direct-risk and risk-cue lexicons used by IGA and the
# lexical verifier; categories match the shared-state risk hypotheses in Sec. 3.2.
PROMPT_RISK_LEXICON_PATH = Path(__file__).with_name("prompt_risk.json")

# Paper Sec. 3.7(b): explicit/weak visual cues used by VAA's lexical channel.
VISUAL_RISK_LEXICON_PATH = Path(__file__).with_name("visual_risk.json")

ATTACK_MARKERS_PATH = Path(__file__).with_name("attack_makers.json")

NSFW_BLACKLIST_PATH = Path(__file__).with_name("nsfw_blacklist_terms.json")

NSFW_MINOR_BOUNDARY_CUES = ("18+", "barely legal")
NSFW_EXPLOITATION_CUES = ("date rape", "rape", "raped", "raper")

_CLIP_TEXT_SINGLETON: Dict[str, Any] = {}


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def term_in_text(term: str, text: str) -> bool:
    term = normalize_text(term.lower())
    if not term:
        return False
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    if re.fullmatch(r"[a-z0-9][a-z0-9\s|._<>/-]*[a-z0-9]", term):
        return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None
    return term in text


def nonempty_items(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if value:
        return [str(value).strip()]
    return []


def meaningful_token_overlap(left: str, right: str) -> set[str]:
    left_tokens = {token for token in token_set(left) if len(token) >= 4}
    right_tokens = {token for token in token_set(right) if len(token) >= 4}
    overlap = left_tokens & right_tokens
    return overlap


def risk_lexicon_scan(text: str, terms_by_category: Dict[str, Sequence[str]]) -> Dict[str, Any]:
    lowered = text.lower()
    categories: List[str] = []
    cues: List[str] = []
    score = 0
    for category, terms in terms_by_category.items():
        hits = [term for term in terms if term_in_text(term, lowered)]
        if hits:
            categories.append(category)
            cues.extend(hits)
            score += len(hits)
    intense = (
        "blood",
        "body horror",
        "corpse",
        "genitals",
        "gore",
        "hitler",
        "nude",
        "nudity",
        "severed",
        "swastika",
        "torture",
    )
    score += sum(2 for cue in cues if cue in intense)
    return {
        "categories": sorted(set(categories)),
        "cues": sorted(set(cues)),
        "score": min(5, score),
    }


def _load_clip_text_encoder(model_path: str) -> Dict[str, Any]:
    cached = _CLIP_TEXT_SINGLETON.get(model_path)
    if cached:
        return cached
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
    model = CLIPModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    if device == "cuda":
        model = model.to(device).half()
    else:
        model = model.to(device)
    cached = {"tokenizer": tokenizer, "model": model, "device": device}
    _CLIP_TEXT_SINGLETON[model_path] = cached
    return cached


def clip_text_cosine(left: str, right: str, model_path: str) -> float:
    import torch
    import torch.nn.functional as F

    bundle = _load_clip_text_encoder(model_path)
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    inputs = tokenizer(
        [left, right],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = F.normalize(features.float(), dim=-1)
        return float((features[0] * features[1]).sum().item())


def load_nsfw_blacklist_terms() -> List[str]:
    if not NSFW_BLACKLIST_PATH.exists():
        return []
    try:
        payload = json.loads(NSFW_BLACKLIST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    terms = payload.get("terms", []) if isinstance(payload, dict) else []
    cleaned: List[str] = []
    seen = set()
    for term in terms:
        normalized = normalize_text(str(term).lower())
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


@dataclass
class ToolFinding:
    tool: str
    decision: str
    category: str
    severity: str
    confidence: float
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "decision": self.decision,
            "category": self.category,
            "severity": self.severity,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


@dataclass
class SemanticAnalysis:
    normalized_prompt: str
    risk_score: int
    categories: List[str]
    explicit_cues: List[str]
    suggestive_cues: List[str]
    attack_markers: List[str]
    reasoning_steps: List[str]
    should_rewrite: bool
    rewrite_directives: List[str] = field(default_factory=list)
    forbidden_semantics: List[str] = field(default_factory=list)
    backend: str = "qwen"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "normalized_prompt": self.normalized_prompt,
            "risk_score": self.risk_score,
            "categories": list(self.categories),
            "explicit_cues": list(self.explicit_cues),
            "suggestive_cues": list(self.suggestive_cues),
            "attack_markers": list(self.attack_markers),
            "reasoning_steps": list(self.reasoning_steps),
            "should_rewrite": self.should_rewrite,
            "rewrite_directives": list(self.rewrite_directives),
            "forbidden_semantics": list(self.forbidden_semantics),
            "backend": self.backend,
        }


CASR_SLOT_ROLES = ("subject", "action", "setting", "style", "modifiers", "misc")


@dataclass
class CASRSlot:
    role: str
    text: str
    risk: str = "safe"
    replacement: str = ""
    score: float = 0.0
    bucket: str = "safe"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "text": self.text,
            "risk": self.risk,
            "replacement": self.replacement,
            "score": round(float(self.score), 4),
            "bucket": self.bucket,
        }


@dataclass
class CASRDecomposition:
    slots: List[CASRSlot]
    reassembled_l1: str
    backend: str = "qwen"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slots": [slot.to_dict() for slot in self.slots],
            "reassembled_l1": self.reassembled_l1,
            "backend": self.backend,
        }


class KeywordPolicyTool:
    def __init__(self) -> None:
        self.nsfw_terms = load_nsfw_blacklist_terms()

    def evaluate(self, prompt: str) -> List[ToolFinding]:
        lowered = prompt.lower()
        findings: List[ToolFinding] = []
        for category, terms in EXPLICIT_TERMS.items():
            hits = [term for term in terms if self._contains_term(lowered, term)]
            if hits:
                findings.append(
                    ToolFinding(
                        tool="keyword_policy",
                        decision="reject",
                        category=category,
                        severity="high",
                        confidence=1.0,
                        reasons=[f"explicit policy terms: {', '.join(hits[:8])}"],
                        metadata={"hits": hits},
                    )
                )
        findings.extend(self._evaluate_nsfw_blacklist(lowered))
        return findings

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        term = normalize_text(term.lower())
        if " " in term:
            return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None

    def _evaluate_nsfw_blacklist(self, text: str) -> List[ToolFinding]:
        if not self.nsfw_terms:
            return []
        high_risk: List[str] = []
        contextual: List[str] = []
        for term in self.nsfw_terms:
            if not self._contains_blacklist_term(text, term):
                continue
            if self._is_high_risk_nsfw(term):
                high_risk.append(term)
            else:
                contextual.append(term)

        findings: List[ToolFinding] = []
        if high_risk:
            findings.append(
                ToolFinding(
                    tool="nsfw_blacklist_high_risk",
                    decision="reject",
                    category=self._high_risk_category(high_risk),
                    severity="high",
                    confidence=1.0,
                    reasons=[f"high-risk NSFW blacklist terms: {', '.join(high_risk[:12])}"],
                    metadata={
                        "hits": high_risk,
                        "match_policy": "exact token/phrase boundary match; wildcard terms are token-local patterns",
                    },
                )
            )
        if contextual:
            category = "hate" if any(self._is_hate_or_abuse(term) for term in contextual) else "jailbreak"
            findings.append(
                ToolFinding(
                    tool="nsfw_blacklist_context",
                    decision="suspicious",
                    category=category,
                    severity="medium",
                    confidence=min(0.9, 0.35 + 0.08 * len(contextual)),
                    reasons=[f"contextual profanity/abuse blacklist terms: {', '.join(contextual[:12])}"],
                    metadata={
                        "hits": contextual,
                        "match_policy": "exact token/phrase boundary match; intended for downstream attention and cleanup",
                    },
                )
            )
        return findings

    @staticmethod
    def _contains_blacklist_term(text: str, term: str) -> bool:
        term = normalize_text(term.lower())
        if not term:
            return False
        if "*" in term:
            pattern = re.escape(term).replace(r"\*", r"[a-z0-9_-]*")
            return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text) is not None
        if " " in term:
            pattern = r"\s+".join(re.escape(part) for part in term.split())
            return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text) is not None
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None

    @staticmethod
    def _is_high_risk_nsfw(term: str) -> bool:
        return any(cue in term for cue in NSFW_HIGH_RISK_CUES)

    @staticmethod
    def _is_hate_or_abuse(term: str) -> bool:
        return any(cue in term for cue in NSFW_HATE_OR_ABUSE_CUES)

    @staticmethod
    def _high_risk_category(terms: Sequence[str]) -> str:
        lowered = " | ".join(terms)
        if any(cue in lowered for cue in NSFW_MINOR_BOUNDARY_CUES):
            return "minors"
        if any(cue in lowered for cue in NSFW_EXPLOITATION_CUES):
            return "exploitation"
        return "sexual"


class GarbledStructureTool:
    def evaluate(self, prompt: str) -> ToolFinding | None:
        if not prompt:
            return None
        chars = len(prompt)
        symbol_count = sum(1 for ch in prompt if not ch.isalnum() and not ch.isspace())
        non_ascii = sum(1 for ch in prompt if ord(ch) > 127)
        long_tokens = [tok for tok in prompt.split() if len(tok) >= 22]
        marker_hits = [marker for marker in ATTACK_MARKERS if marker.lower() in prompt.lower()]
        symbol_ratio = symbol_count / max(chars, 1)
        non_ascii_ratio = non_ascii / max(chars, 1)
        score = 0.0
        score += 2.0 if symbol_ratio > 0.35 else 0.0
        score += 1.5 if non_ascii_ratio > 0.08 else 0.0
        score += min(2.0, len(long_tokens) * 0.75)
        score += len(marker_hits) * 1.5
        if score < 1.5:
            return None
        decision = "reject" if score >= 4.0 else "suspicious"
        severity = "high" if decision == "reject" else "medium"
        return ToolFinding(
            tool="garbled_structure",
            decision=decision,
            category="jailbreak",
            severity=severity,
            confidence=min(1.0, score / 5.0),
            reasons=[
                f"symbol_ratio={round(symbol_ratio, 3)}",
                f"non_ascii_ratio={round(non_ascii_ratio, 3)}",
                f"long_tokens={len(long_tokens)}",
                f"attack_markers={marker_hits}",
            ],
            metadata={
                "score": score,
                "symbol_ratio": symbol_ratio,
                "non_ascii_ratio": non_ascii_ratio,
                "long_tokens": long_tokens[:8],
                "marker_hits": marker_hits,
            },
        )


class SuggestiveCueTool:
    def evaluate(self, prompt: str) -> List[ToolFinding]:
        lowered = prompt.lower()
        findings: List[ToolFinding] = []
        for category, terms in SUGGESTIVE_TERMS.items():
            hits = [term for term in terms if term_in_text(term, lowered)]
            if hits:
                findings.append(
                    ToolFinding(
                        tool="suggestive_cues",
                        decision="suspicious",
                        category=category,
                        severity="medium",
                        confidence=min(0.85, 0.35 + 0.12 * len(hits)),
                        reasons=[f"suggestive cues: {', '.join(hits[:8])}"],
                        metadata={"hits": hits},
                    )
                )
        return findings


# IGA performs rapid pre-generation triage of explicit risks, adversarial
# structures, and risk cues.
class IGAgent:
    def __init__(self, llm: QwenLLMClient, config: AgentVersionConfig):
        self.llm = llm
        self.config = config
        self.keyword_tool = KeywordPolicyTool()
        self.garbled_tool = GarbledStructureTool()
        self.suggestive_tool = SuggestiveCueTool()

    def run(self, state: DefenseState) -> DefenseState:
        # Paper Sec. 3.5: low-latency lexical, structural, and optional
        # lightweight-LLM triage before any image generation.
        start = time.perf_counter()
        normalized = normalize_text(state.original_prompt)
        state.normalized_prompt = normalized
        state.current_prompt = normalized

        findings: List[ToolFinding] = []
        findings.extend(self.keyword_tool.evaluate(normalized))
        garbled = self.garbled_tool.evaluate(state.original_prompt)
        if garbled:
            findings.append(garbled)
        findings.extend(self.suggestive_tool.evaluate(normalized))

        state.tool_outputs["ig_agent_tools"] = [finding.to_dict() for finding in findings]
        for finding in findings:
            state.add_evidence(
                AgentEvidence(
                    source="ig_agent",
                    category=finding.category,
                    severity=finding.severity,
                    confidence=finding.confidence,
                    text="; ".join(finding.reasons),
                    metadata=finding.to_dict(),
                )
            )

        has_reject = any(finding.decision == "reject" for finding in findings)
        has_suspicious = any(finding.decision == "suspicious" for finding in findings)
        state.iga_status = "reject" if has_reject else "handoff_to_ssa" if has_suspicious else "pass"

        if self.config.use_ig_llm_triage and not has_reject:
            triage = self._llm_triage(state)
            if triage:
                state.tool_outputs["ig_agent_llm_triage"] = triage
                self._merge_llm_triage(state, triage)

        if state.iga_status == "reject" and not self.config.ig_allow_block:
            state.tool_outputs["ig_agent_report_only"] = {
                "original_status": "reject",
                "reason": "ig_allow_block=false; preserving all IG evidence and routing to SSA",
            }
            state.iga_status = "handoff_to_ssa"

        if state.iga_status == "reject":
            state.add_route("ig_agent:block")
        elif state.iga_status == "handoff_to_ssa":
            state.add_route("ig_agent:handoff_to_ssa")
        else:
            state.add_route("ig_agent:pass")
        state.timings["ig_agent"] = round(time.perf_counter() - start, 4)
        return state

    def _llm_triage(self, state: DefenseState) -> Dict[str, Any]:
        system = (
            "You are the Input Guardrail Agent (IGA) responsible for rapid pre-generation triage "
            "of text-to-image prompts. The rule-tool results are already provided; make only a "
            "quick supplementary judgment and return strict JSON."
        )
        user = f"""PROMPT:
{state.original_prompt}

RULE-TOOL EVIDENCE:
{state.risk_summary()}

OUTPUT JSON:
{{
  "decision": "pass | handoff_to_ssa | reject",
  "risk_hypotheses": ["sexual|violence|self-harm|hate|drugs|minors|exploitation|jailbreak"],
  "evidence": [{{"category": "...", "severity": "low|medium|high", "confidence": 0.0, "text": "..."}}],
  "reason": "one concise English reason"
}}

BOUNDARY: explicit violations or clear attack structures may be rejected; implicit, associative, or semantically uncertain risks should be handed off to SSA."""
        return self.llm.chat_json(system, user, max_tokens=500)


# SSA analyzes latent semantic intent and selects an unchanged, CASR, or block
# path according to the identified risk.
class SSAgent:
    def __init__(self, llm: QwenLLMClient, config: AgentVersionConfig):
        self.llm = llm
        self.config = config

    def run(self, state: DefenseState) -> DefenseState:
        # Paper Sec. 3.6: semantic intent analysis selects pass, CASR rewrite,
        # or block when a safe rewrite cannot be verified.
        start = time.perf_counter()
        payload = self._llm_analyze(state)
        analysis = self._payload_to_analysis(state, payload)
        if analysis is None:
            state.tool_outputs["ss_agent_analysis"] = {
                "backend": "qwen",
                "error": "LLM semantic analysis did not return a complete structured result",
            }
            state.ssa_status = "reject_llm_analysis"
            state.final_decision = "block"
            state.block_stage = "ss_agent"
            state.rejection_reason = "SSA requires a complete LLM semantic analysis"
            state.add_route("ss_agent:block_llm_analysis")
            state.timings["ss_agent"] = round(time.perf_counter() - start, 4)
            return state
        memory_context = self._augment_with_failure_memory(state, analysis)
        state.tool_outputs["ss_agent_failure_memory_analysis"] = memory_context
        state.tool_outputs["ss_agent_analysis"] = analysis.to_dict()
        for category in analysis.categories:
            state.add_evidence(
                AgentEvidence(
                    source="ss_agent",
                    category=category,
                    severity="high" if analysis.risk_score >= 4 else "medium",
                    confidence=min(1.0, max(0.1, analysis.risk_score / 5.0)),
                    text="; ".join((analysis.explicit_cues + analysis.suggestive_cues + analysis.attack_markers)[:6]),
                    metadata={"risk_score": analysis.risk_score},
                )
            )

        if not analysis.should_rewrite and not analysis.attack_markers:
            state.ssa_status = "pass_original"
            state.add_route("ss_agent:pass_original")
            state.timings["ss_agent"] = round(time.perf_counter() - start, 4)
            return state

        if self.config.ablate_casr:
            self._run_ablate_casr(state, analysis)
            state.timings["ss_agent"] = round(time.perf_counter() - start, 4)
            return state
        decomp = self._casr_decompose_and_rewrite(state, analysis)
        self._score_casr_slots(decomp, state)
        state.tool_outputs["ss_agent_casr_decomposition"] = decomp.to_dict()
        verifier_l1, lexical_verify, k_fail_l1 = self._verify_casr_candidate(state, analysis, decomp.reassembled_l1, decomp.slots)
        state.tool_outputs["ss_agent_verifier"] = verifier_l1
        if self._casr_is_safe(verifier_l1):
            state.ssa_status = "sanitize"
            state.sanitized_prompt = decomp.reassembled_l1
            state.current_prompt = decomp.reassembled_l1
            state.add_route("ss_agent:sanitize")
            state.timings["ss_agent"] = round(time.perf_counter() - start, 4)
            return state

        l2 = self._casr_l2_abstract_failed_slots(decomp.reassembled_l1, k_fail_l1, decomp.slots, analysis)
        p_l2 = normalize_text(str(l2.get("reassembled_l2") or ""))
        if not p_l2:
            state.tool_outputs["ss_agent_casr_l2"] = {
                **l2,
                "reassembled_l2": "",
                "backend": str(l2.get("backend") or "empty_l2"),
                "reason": "CASR L2 did not produce a candidate",
            }
            self._write_casr_lesson(state, analysis, decomp)
            state.ssa_status = "reject_failed_sanitize"
            state.final_decision = "block"
            state.block_stage = "ss_agent"
            state.rejection_reason = "CASR L1 failed and L2 did not produce a safe abstraction"
            state.add_route("ss_agent:block_failed_sanitize")
            state.timings["ss_agent"] = round(time.perf_counter() - start, 4)
            return state
        state.tool_outputs["ss_agent_casr_l2"] = l2
        verifier_l2, _lexical_verify, _k_fail_l2 = self._verify_casr_candidate(state, analysis, p_l2, decomp.slots)
        state.tool_outputs["ss_agent_verifier_after_second_rewrite"] = verifier_l2
        if self._casr_is_safe(verifier_l2):
            state.ssa_status = "sanitize"
            state.sanitized_prompt = p_l2
            state.current_prompt = p_l2
            state.add_route("ss_agent:sanitize")
        else:
            self._write_casr_lesson(state, analysis, decomp)
            state.ssa_status = "reject_failed_sanitize"
            state.final_decision = "block"
            state.block_stage = "ss_agent"
            state.rejection_reason = str(verifier_l2.get("reason") or "high risk prompt could not be safely sanitized")
            state.add_route("ss_agent:block_failed_sanitize")

        state.timings["ss_agent"] = round(time.perf_counter() - start, 4)
        return state

    def _run_ablate_casr(
        self,
        state: DefenseState,
        analysis: SemanticAnalysis,
    ) -> None:
        candidate = normalize_text(state.current_prompt or state.original_prompt)
        llm_rewrite = self._llm_rewrite(state, analysis, candidate)
        sanitized = normalize_text(str(llm_rewrite.get("sanitized_prompt") or ""))
        state.tool_outputs["ss_agent_llm_rewrite"] = {
            **llm_rewrite,
            "method": "ablate_casr_single_llm_rewrite",
        }
        if not sanitized:
            state.ssa_status = "reject_failed_sanitize"
            state.final_decision = "block"
            state.block_stage = "ss_agent"
            state.rejection_reason = "CASR ablation requires an LLM rewrite candidate"
            state.add_route("ss_agent:block_invalid_llm_rewrite")
            return

        verifier = self._verify_rewrite(state, analysis, sanitized)
        state.tool_outputs["ss_agent_verifier"] = verifier
        lexical_verify = self._lexical_risk_verify_prompt(sanitized)
        state.tool_outputs["ss_agent_lexical_risk_verifier"] = lexical_verify
        is_safe = bool(verifier.get("is_safe", True)) and bool(lexical_verify.get("is_safe", True))

        if (
            not is_safe
            and self.config.block_on_failed_high_risk_sanitize
            and analysis.risk_score >= self.config.high_risk_reject_threshold
        ):
            state.ssa_status = "reject_failed_sanitize"
            state.final_decision = "block"
            state.block_stage = "ss_agent"
            state.rejection_reason = str(
                verifier.get("reason")
                or lexical_verify.get("reason")
                or "single rewrite could not be safely sanitized"
            )
            state.add_route("ss_agent:block_failed_sanitize")
        else:
            state.ssa_status = "sanitize"
            state.sanitized_prompt = sanitized
            state.current_prompt = sanitized
            state.add_route("ss_agent:sanitize")

    def _augment_with_failure_memory(self, state: DefenseState, analysis: SemanticAnalysis) -> Dict[str, Any]:
        # Paper Sec. 3.6.1: retrieved lessons augment categories, forbidden
        # semantics, and rewrite guidance; a match forces CASR processing.
        if not self.config.use_ss_failure_memory:
            return {
                "categories": [],
                "cues": [],
                "score": 0,
                "use_strong_rewrite": False,
                "matched_count": 0,
                "disabled": True,
            }
        memory_context = state.tool_outputs.get("failure_memory_context", {})
        lessons = memory_context.get("lessons", []) if isinstance(memory_context, dict) else []
        if not isinstance(lessons, list) or not lessons:
            return {
                "categories": [],
                "cues": [],
                "score": 0,
                "use_strong_rewrite": False,
                "matched_count": 0,
            }
        categories: List[str] = []
        cues: List[str] = []
        actions: List[str] = []
        for lesson in lessons[: self.config.failure_memory_max_lessons]:
            if not isinstance(lesson, dict):
                continue
            categories.extend(str(cat) for cat in lesson.get("categories", []) if str(cat) in RISK_CATEGORIES)
            cues.extend(str(cue) for cue in lesson.get("matched_prompt_cues", []) if str(cue).strip())
            cues.extend(str(cue) for cue in lesson.get("prompt_cues", [])[:8] if str(cue).strip())
            actions.extend(str(action) for action in lesson.get("recommended_actions", []) if str(action).strip())
        categories = sorted(set(categories)) or ["jailbreak"]
        cues = sorted(set(cues))[:32]
        for category in categories:
            if category not in analysis.categories:
                analysis.categories.append(category)
        analysis.suggestive_cues = sorted(set(analysis.suggestive_cues + cues))
        analysis.forbidden_semantics = sorted(set(analysis.forbidden_semantics + cues))
        analysis.rewrite_directives = sorted(
            set(
                analysis.rewrite_directives
                + actions[:6]
                + [
                    "previous official-judge failure patterns matched; do not pass the original prompt",
                    "use CASR L1/L2 slot abstraction only; block if the risky slots cannot be cleaned",
                ]
            )
        )
        analysis.risk_score = max(analysis.risk_score, 4)
        analysis.should_rewrite = True
        if "failure_memory" not in analysis.backend:
            analysis.backend = f"{analysis.backend}+failure_memory"
        return {
            "categories": categories,
            "cues": cues,
            "score": min(5, 2 + len(cues)),
            "use_strong_rewrite": True,
            "matched_count": len(lessons),
        }

    def _llm_analyze(self, state: DefenseState) -> Dict[str, Any]:
        system = (
            "You are the Semantic Safety Agent (SSA) responsible for latent semantic intent analysis, "
            "risk hypotheses, and sanitization decisions for text-to-image prompts. Use the IGA "
            "evidence in the shared state to identify explicit risks, implicit associative risks, "
            "and jailbreak structures. Based on the risk categories, explicit or suggestive risk "
            "evidence, and attack structures, decide whether safe rewriting is required. Return "
            "strict JSON without Markdown."
        )
        user = f"""PROMPT:
{state.current_prompt or state.original_prompt}

SHARED-STATE EVIDENCE:
{state.risk_summary()}

OUTPUT JSON:
{{
  "normalized_prompt": "string",
  "risk_score": 0,
  "categories": ["sexual|violence|self-harm|hate|drugs|minors|exploitation|jailbreak"],
  "explicit_cues": ["..."],
  "suggestive_cues": ["..."],
  "attack_markers": ["..."],
  "reasoning_steps": ["4 to 6 concise English reasoning statements"],
  "should_rewrite": true,
  "rewrite_directives": ["actions that must be performed during sanitization"],
  "forbidden_semantics": ["dangerous semantics that must not remain after sanitization"]
}}"""
        return self.llm.chat_json(system, user, max_tokens=900)

    def _payload_to_analysis(self, state: DefenseState, payload: Dict[str, Any]) -> SemanticAnalysis | None:
        required = {
            "normalized_prompt",
            "risk_score",
            "categories",
            "explicit_cues",
            "suggestive_cues",
            "attack_markers",
            "reasoning_steps",
            "should_rewrite",
            "rewrite_directives",
            "forbidden_semantics",
        }
        if not isinstance(payload, dict) or not required.issubset(payload):
            return None
        explicit_cues = [str(x) for x in payload.get("explicit_cues", []) if str(x).strip()]
        suggestive_cues = [str(x) for x in payload.get("suggestive_cues", []) if str(x).strip()]
        attack_markers = [str(x) for x in payload.get("attack_markers", []) if str(x).strip()]
        risk_score = int(payload.get("risk_score", 0) or 0)
        categories = [str(item) for item in payload.get("categories", []) if str(item) in RISK_CATEGORIES]
        if risk_score <= 0 and not explicit_cues and not suggestive_cues and not attack_markers:
            categories = []
            should_rewrite = False
        else:
            should_rewrite = bool(payload.get("should_rewrite", False))
        return SemanticAnalysis(
            normalized_prompt=str(payload.get("normalized_prompt") or normalize_text(state.current_prompt or state.original_prompt)),
            risk_score=risk_score,
            categories=categories,
            explicit_cues=explicit_cues,
            suggestive_cues=suggestive_cues,
            attack_markers=attack_markers,
            reasoning_steps=[str(x) for x in payload.get("reasoning_steps", []) if str(x).strip()],
            should_rewrite=should_rewrite,
            rewrite_directives=[str(x) for x in payload.get("rewrite_directives", []) if str(x).strip()],
            forbidden_semantics=[str(x) for x in payload.get("forbidden_semantics", []) if str(x).strip()],
            backend="qwen",
        )


    def _casr_decompose_and_rewrite(self, state: DefenseState, analysis: SemanticAnalysis) -> CASRDecomposition:
        # Paper Sec. 3.6.2 (CASR L1): one structured LLM call decomposes the
        # prompt into six semantic slots and reassembles a candidate prompt.
        prompt = normalize_text(state.current_prompt or state.original_prompt)
        system = (
            "You are the component-aware sanitization and rewriting module of SSA. Decompose the "
            "input prompt into six semantic-component slots:\n"
            "  subject, action, setting, medium and artistic style, affective modifiers, and residual components.\n\n"
            "For each slot provide:\n"
            '  - text: the original fragment\n  - risk: "safe" | "partial" | "risky"\n'
            "      safe = contains no risk semantics\n"
            "      partial = contains some risk cues that can be locally removed\n"
            "      risky = the slot semantics violate the sanitization objective and require replacement\n"
            "  - replacement: when risk != safe, provide a replacement that must satisfy:\n"
            "      (a) preserve the grammatical function (subject -> subject)\n"
            '      (b) preserve any retainable abstract semantic class ("a person" remains "a person")\n'
            "      (c) introduce no new concrete entities\n"
            "      (d) trigger no risk lexicon terms\n\n"
            "Return strict JSON:\n"
            '{\n  "slots": [\n'
            '    {"role": "subject|action|setting|style|modifiers|misc",\n'
            '     "text": "...", "risk": "safe|partial|risky", "replacement": "..."},\n'
            "    ...\n  ],\n"
            '  "reassembled_l1": "the grammatically coherent L1 candidate prompt"\n}'
        )
        user = f"""INPUT PROMPT:
{prompt}

RISK ANALYSIS:
{analysis.to_dict()}

Return strict JSON."""
        payload = self.llm.chat_json(system, user, max_tokens=900)
        decomp = self._payload_to_casr_decomposition(payload)
        if decomp.slots and decomp.reassembled_l1:
            return decomp
        return CASRDecomposition([], "", backend="qwen_invalid")

    def _payload_to_casr_decomposition(self, payload: Dict[str, Any]) -> CASRDecomposition:
        if not isinstance(payload, dict):
            return CASRDecomposition([], "", backend="empty")
        slots: List[CASRSlot] = []
        for item in payload.get("slots", []) if isinstance(payload.get("slots"), list) else []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "misc")).strip().lower()
            if role not in CASR_SLOT_ROLES:
                role = "misc"
            text = normalize_text(str(item.get("text") or ""))
            replacement = normalize_text(str(item.get("replacement") or ""))
            risk = str(item.get("risk", "safe")).strip().lower()
            if risk not in {"safe", "partial", "risky"}:
                risk = "safe"
            if text:
                slots.append(CASRSlot(role=role, text=text, risk=risk, replacement=replacement))
        reassembled = normalize_text(str(payload.get("reassembled_l1") or ""))
        return CASRDecomposition(slots=slots, reassembled_l1=reassembled, backend="qwen")


    def _score_casr_slots(self, decomp: CASRDecomposition, state: DefenseState) -> None:
        # Paper Eq. (1), Sec. 3.6.2: s(slot) = alpha*s_lex + beta*s_sem +
        # gamma*s_mem, followed by safe/partial/risky threshold bucketing.
        memory_context = state.tool_outputs.get("failure_memory_context", {})
        lessons = memory_context.get("lessons", []) if isinstance(memory_context, dict) else []
        for slot in decomp.slots:
            lex = self._slot_lex_score(slot.text)
            sem = {"safe": 0.0, "partial": 0.5, "risky": 1.0}.get(slot.risk, 0.0)
            mem = self._slot_memory_score(slot, lessons if isinstance(lessons, list) else [])
            score = (
                float(self.config.casr_alpha) * lex
                + float(self.config.casr_beta) * sem
                + float(self.config.casr_gamma) * mem
            )
            slot.score = max(0.0, min(1.0, score))
            if slot.score < float(self.config.casr_tau_safe):
                slot.bucket = "safe"
            elif slot.score >= float(self.config.casr_tau_risky):
                slot.bucket = "risky"
            else:
                slot.bucket = "partial"

    @staticmethod
    def _slot_lex_score(text: str) -> float:
        scan = risk_lexicon_scan(text, PROMPT_RISK_LEXICON)
        return min(1.0, float(scan.get("score", 0) or 0) / 5.0)

    @staticmethod
    def _slot_memory_score(slot: CASRSlot, lessons: List[Any]) -> float:
        slot_tokens = token_set(slot.text)
        if not slot_tokens:
            return 0.0
        best = 0.0
        for lesson in lessons:
            if not isinstance(lesson, dict):
                continue
            cues = lesson.get("slot_cues", [])
            if not isinstance(cues, list):
                continue
            for cue in cues:
                cue_text = str(cue)
                if ":" in cue_text:
                    role, cue_body = cue_text.split(":", 1)
                    if role.strip() and role.strip() != slot.role:
                        continue
                    cue_text = cue_body
                cue_tokens = token_set(cue_text)
                if cue_tokens:
                    best = max(best, len(slot_tokens & cue_tokens) / len(slot_tokens | cue_tokens))
        return best

    def _verify_casr_candidate(
        self,
        state: DefenseState,
        analysis: SemanticAnalysis,
        candidate: str,
        slots: List[CASRSlot],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[int]]:
        # Paper Sec. 3.6.3: LLM and lexical verifiers run together; K_fail
        # identifies residual risky slots for the L2 abstraction pass.
        verifier = self._verify_rewrite(state, analysis, candidate)
        lexical_verify = self._lexical_risk_verify_prompt(candidate)
        k_fail: List[int] = []
        for idx, slot in enumerate(slots):
            text_after = slot.replacement if slot.risk != "safe" and slot.replacement else slot.text
            if not self._lexical_risk_verify_prompt(text_after)["is_safe"]:
                k_fail.append(idx)
        if not k_fail and not lexical_verify.get("is_safe", True):
            remaining = [str(cue).lower() for cue in lexical_verify.get("remaining_lexical_cues", [])]
            for idx, slot in enumerate(slots):
                text_after = (slot.replacement if slot.risk != "safe" and slot.replacement else slot.text).lower()
                if any(cue and term_in_text(cue, text_after) for cue in remaining):
                    k_fail.append(idx)
            if not k_fail:
                k_fail = [idx for idx, slot in enumerate(slots) if slot.bucket != "safe"] or list(range(len(slots)))
        if self._verifier_reports_risk(verifier):
            remaining_text = " ".join(item.lower() for item in nonempty_items(verifier.get("remaining_risks", [])))
            remaining_text = remaining_text or str(verifier.get("reason", "")).lower()
            for idx, slot in enumerate(slots):
                text_after = (slot.replacement if slot.risk != "safe" and slot.replacement else slot.text).lower()
                original = slot.text.lower()
                has_exact_match = (
                    (text_after and term_in_text(text_after, remaining_text))
                    or (original and term_in_text(original, remaining_text))
                )
                has_token_overlap = bool(
                    meaningful_token_overlap(remaining_text, text_after)
                    or meaningful_token_overlap(remaining_text, original)
                )
                if has_exact_match or has_token_overlap:
                    if idx not in k_fail:
                        k_fail.append(idx)
            if not k_fail:
                k_fail = [idx for idx, slot in enumerate(slots) if slot.bucket != "safe" or slot.risk != "safe"] or list(range(len(slots)))
        lexical_verify["K_fail"] = k_fail
        return verifier, lexical_verify, k_fail

    @staticmethod
    def _verifier_reports_risk(verifier: Dict[str, Any]) -> bool:
        remaining = nonempty_items(verifier.get("remaining_risks", []))
        return (not bool(verifier.get("is_safe", True))) or bool(verifier.get("needs_second_rewrite", False)) or bool(remaining)

    @staticmethod
    def _casr_is_safe(verifier: Dict[str, Any]) -> bool:
        verifier_safe = (
            bool(verifier.get("is_safe", True))
            and not bool(verifier.get("needs_second_rewrite", False))
            and not nonempty_items(verifier.get("remaining_risks", []))
        )
        return verifier_safe

    def _casr_l2_abstract_failed_slots(
        self,
        p_l1: str,
        k_fail: List[int],
        slots: List[CASRSlot],
        analysis: SemanticAnalysis,
    ) -> Dict[str, Any]:
        # Paper Sec. 3.6.2 (CASR L2): abstract only failed slots to broader,
        # neutral concepts while preserving slots that passed L1.
        if not k_fail:
            k_fail = [idx for idx, slot in enumerate(slots) if slot.bucket != "safe"]
        slot_texts = [
            {
                "index": idx,
                "role": slots[idx].role,
                "text": slots[idx].replacement if slots[idx].risk != "safe" and slots[idx].replacement else slots[idx].text,
            }
            for idx in k_fail
            if 0 <= idx < len(slots)
        ]
        system = (
            "You are SSA's component-level abstraction module. Given:\n"
            f"  - L1 candidate prompt: {p_l1}\n"
            f"  - slot indices still triggering the risk lexicon according to verification: {k_fail}\n"
            f"  - the L1 text of these slots: {slot_texts}\n\n"
            'Apply "abstraction" only to slots in K_fail:\n'
            '  - generalize a concrete concept to a broader safe superordinate category\n'
            "  - introduce no new concrete entities\n"
            "  - trigger no risk lexicon terms\n"
            "  - remove residual risk modifiers identified by verification; if no safe candidate can be generated, return an empty reassembled_l2\n"
            "Keep the text of all slots that passed L1 unchanged.\n\n"
            "Return strict JSON:\n"
            '{\n  "abstracted_slots": [{"role": "...", "old": "...", "new": "..."}, ...],\n'
            '  "reassembled_l2": "..."\n}'
        )
        user = f"""ORIGINAL RISK ANALYSIS:
{analysis.to_dict()}

Return strict JSON."""
        payload = self.llm.chat_json(system, user, max_tokens=700)
        if isinstance(payload, dict) and normalize_text(str(payload.get("reassembled_l2") or "")):
            payload.setdefault("K_fail", k_fail)
            payload.setdefault("slot_texts", slot_texts)
            payload.setdefault("backend", "qwen")
            return payload

        return {
            "K_fail": k_fail,
            "slot_texts": slot_texts,
            "abstracted_slots": [],
            "reassembled_l2": "",
            "backend": "qwen_invalid",
            "reason": "CASR L2 did not return a complete LLM candidate",
        }

    def _write_casr_lesson(self, state: DefenseState, analysis: SemanticAnalysis, decomp: CASRDecomposition) -> None:
        # Paper Sec. 3.3: an L1/L2 sanitization failure becomes a reusable
        # failure-memory lesson containing residual risky slot cues.
        if self.config.failure_memory_read_only:
            state.tool_outputs["ss_agent_casr_lesson"] = {
                "path": str(self.config.failure_memory_path),
                "skipped": "failure_memory_read_only",
            }
            return
        try:
            from .failure_memory import update_lessons

            slot_cues = [
                f"{slot.role}:{slot.text}"
                for slot in decomp.slots
                if slot.text and (slot.bucket != "safe" or slot.risk != "safe")
            ]
            case = {
                "source_run": "casr",
                "i2p_index": state.request_id,
                "judge_failure": "casr_failed_sanitize",
                "categories": analysis.categories,
                "prompt_cues": sorted(set(analysis.explicit_cues + analysis.suggestive_cues + analysis.attack_markers))[:24],
                "visual_cues": [],
                "slot_cues": slot_cues[:32],
                "recommended_actions": ["CASR failed; block similar risky slot combinations before generation."],
                "failure_reason": "CASR L1 and L2 verification failed.",
                "original_prompt": state.original_prompt,
                "current_prompt": state.current_prompt,
                "visual_description": "",
                "image_path": state.image_path,
            }
            memory = update_lessons(self.config.failure_memory_path, case)
            state.tool_outputs["ss_agent_casr_lesson"] = {
                "path": str(self.config.failure_memory_path),
                "slot_cues": slot_cues[:32],
                "lesson_count": len(memory.get("lessons", [])) if isinstance(memory, dict) else 0,
            }
        except Exception as exc:
            state.errors.append(f"SSAgent CASR lesson write failed: {type(exc).__name__}: {exc}")

    def _lexical_risk_verify_prompt(self, prompt: str) -> Dict[str, Any]:
        scan = risk_lexicon_scan(prompt, PROMPT_RISK_LEXICON)
        cues = sorted(set(scan["cues"]))
        categories = list(scan["categories"])
        return {
            "is_safe": not cues,
            "remaining_lexical_cues": cues,
            "categories": sorted(set(categories)),
            "score": min(5, int(scan["score"] or 0)),
            "reason": "lexical risk verifier",
        }

    def _llm_rewrite(
        self,
        state: DefenseState,
        analysis: SemanticAnalysis,
        candidate: str,
        *,
        conservative: bool = False,
    ) -> Dict[str, Any]:
        system = (
            "You are SSA's safe rewriting module. Produce an English text-to-image prompt. "
            "Preserve the safe theme, medium, style, and composition while removing dangerous "
            "semantics, sexualization, humiliating or victimizing relations, self-harm, gore, hate, "
            "drugs, and jailbreak triggers. Return strict JSON."
        )
        mode = "be more conservative and remove risk even if some original semantics are lost" if conservative else "preserve as much safe semantics as possible"
        user = f"""ORIGINAL PROMPT:
{state.original_prompt}

SANITIZED CANDIDATE PROMPT:
{candidate}

RISK ANALYSIS:
{analysis.to_dict()}

Following the instruction to {mode}, return JSON:
{{
  "sanitized_prompt": "safe English text-to-image prompt",
  "removed_cues": ["..."],
  "preserved_safe_semantics": ["..."],
  "reason": "one concise English explanation"
}}"""
        return self.llm.chat_json(system, user, max_tokens=700)

    def _verify_rewrite(self, state: DefenseState, analysis: SemanticAnalysis, sanitized: str) -> Dict[str, Any]:
        system = "You are SSA's rewriting verifier. Judge only whether the sanitized prompt still retains risks from the original prompt. Return strict JSON."
        user = f"""ORIGINAL PROMPT:
{state.original_prompt}

SANITIZED PROMPT:
{sanitized}

ORIGINAL RISK ANALYSIS:
{analysis.to_dict()}

OUTPUT JSON:
{{
  "is_safe": true,
  "remaining_risks": ["..."],
  "needs_second_rewrite": false,
  "reason": "one concise English reason"
}}"""
        payload = self.llm.chat_json(system, user, max_tokens=500)
        return self._postprocess_rewrite_verifier(state, payload, sanitized)

    @staticmethod
    def _postprocess_rewrite_verifier(state: DefenseState, payload: Dict[str, Any], sanitized: str) -> Dict[str, Any]:
        del state, sanitized
        normalized = dict(payload) if isinstance(payload, dict) else {}
        remaining = sorted(set(nonempty_items(normalized.get("remaining_risks", []))))
        is_safe_raw = normalized.get("is_safe", False)
        needs_second_raw = normalized.get("needs_second_rewrite", False)
        is_safe = is_safe_raw is True or str(is_safe_raw).strip().lower() == "true"
        needs_second_rewrite = (
            needs_second_raw is True or str(needs_second_raw).strip().lower() == "true"
        )
        if remaining or needs_second_rewrite:
            is_safe = False
        if not is_safe:
            needs_second_rewrite = True
        normalized["is_safe"] = is_safe
        normalized["remaining_risks"] = remaining
        normalized["needs_second_rewrite"] = needs_second_rewrite
        normalized.setdefault("reason", "LLM rewrite verifier result")
        return normalized


# VAA performs a post-generation safety audit using upstream risk hypotheses
# and three channels: LLM assessment, visual risk lexicon, and failure memory.
class VAAgent:
    def __init__(self, llm: QwenLLMClient, config: AgentVersionConfig):
        self.llm = llm
        self.config = config

    def run(self, state: DefenseState, visual_description: str) -> DefenseState:
        # Paper Sec. 3.7(b)-(d): combine visual lexicon, failure-memory, and LLM
        # channels; any blocking channel prevents image release.
        start = time.perf_counter()
        description = str(visual_description).strip() if self.config.use_va_vlm_description else ""
        state.tool_outputs["va_agent_input_description"] = description
        state.tool_outputs["va_agent_block_channel"] = ""
        state.tool_outputs["va_agent_visual_memory"] = {
            "s_vis_mem": 0.0,
            "threshold": self.config.vaa_tau_vis_mem,
            "decision": "pass",
            "hits": [],
            "skipped": True,
        }
        if not description:
            state.tool_outputs["va_agent_description_disabled"] = not self.config.use_va_vlm_description
            state.vaa_status = "pass"
            state.add_route("va_agent:pass_no_description")
            state.timings["va_agent"] = round(time.perf_counter() - start, 4)
            return state

        vrl_payload = self._visual_risk_lexicon_audit(state, description)
        state.tool_outputs["va_agent_visual_risk_lexicon_audit"] = vrl_payload
        visual_memory = self._visual_memory_score(state, description)
        state.tool_outputs["va_agent_visual_memory"] = visual_memory
        llm_payload = self._llm_audit(state, description)
        signals = {
            "vrl": vrl_payload,
            "vis_mem": visual_memory,
            "llm": llm_payload,
        }
        state.tool_outputs["va_agent_all_channel_signals"] = signals
        channel, payload = self._merge_va_channel_signals(signals)
        state.tool_outputs["va_agent_block_channel"] = channel
        state.tool_outputs["va_agent_audit"] = payload
        decision = str(payload.get("decision", "pass"))
        if decision not in {"pass", "block"}:
            decision = "block" if "block" in decision else "pass"
        state.vaa_status = decision
        if decision == "block":
            categories = payload.get("categories", [])
            category = str(categories[0]) if isinstance(categories, list) and categories else "visual_risk"
            state.add_evidence(
                AgentEvidence(
                    source="va_agent",
                    category=category,
                    severity="high",
                    confidence=float(payload.get("confidence", 0.8) or 0.8),
                    text=str(payload.get("reason", "visual audit blocked")),
                    metadata=payload,
                )
            )
            state.final_decision = "block"
            state.block_stage = "va_agent"
            state.rejection_reason = str(payload.get("reason", "visual audit blocked"))
            state.add_route("va_agent:block")
        else:
            state.add_route("va_agent:pass")
        state.timings["va_agent"] = round(time.perf_counter() - start, 4)
        return state

    def close(self) -> None:
        return None

    def _visual_risk_lexicon_audit(self, state: DefenseState, visual_description: str) -> Dict[str, Any]:
        # Paper Sec. 3.7(b): explicit visual cues block directly, while weak
        # cues block only when consistent with upstream hypotheses H.
        description = normalize_text(visual_description).lower()
        explicit_scan = risk_lexicon_scan(description, VISUAL_RISK_LEXICON)
        weak_scan = risk_lexicon_scan(description, WEAK_VISUAL_RISK_LEXICON)
        upstream = set(state.risk_hypotheses)
        weak_categories = [category for category in weak_scan["categories"] if category in upstream]
        weak_cues = sorted(
            {
                term
                for category in weak_categories
                for term in WEAK_VISUAL_RISK_LEXICON.get(category, ())
                if term_in_text(term, description)
            }
        )
        explicit_cues = list(explicit_scan["cues"])
        explicit_categories = [category for category in explicit_scan["categories"] if category in RISK_CATEGORIES]
        should_block = bool(explicit_cues or weak_cues)
        categories = sorted(set(explicit_categories + weak_categories))
        cues = sorted(set(explicit_cues + weak_cues))
        if explicit_cues:
            reason = "explicit visual risk cue matched the visual risk lexicon"
        elif weak_cues:
            reason = "weak visual risk cue matched an upstream risk hypothesis"
        else:
            reason = "no visual risk lexicon cue matched"
        return {
            "decision": "block" if should_block else "pass",
            "categories": categories,
            "visible_evidence": cues,
            "confidence": 0.85 if explicit_cues else 0.72 if weak_cues else 0.35,
            "reason": reason,
            "explicit_cues": explicit_cues,
            "weak_cues": weak_cues,
            "upstream_risk_hypotheses": list(state.risk_hypotheses),
        }

    @staticmethod
    def _merge_va_channel_signals(signals: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        for channel in ("vrl", "vis_mem", "llm"):
            signal = signals.get(channel, {})
            if signal.get("decision") != "block":
                continue
            payload = dict(signal)
            if channel == "vis_mem":
                payload.setdefault("visible_evidence", signal.get("hits", []))
                payload.setdefault("confidence", 0.82)
                payload.setdefault("reason", "visual description matched previous failure memory")
            payload["channel"] = channel
            return channel, payload
        llm_payload = dict(signals.get("llm", {}))
        if llm_payload:
            llm_payload["decision"] = "pass"
            return "", llm_payload
        return "", {
            "decision": "pass",
            "categories": [],
            "visible_evidence": [],
            "confidence": 0.0,
            "reason": "all VAA channels passed",
        }

    def _llm_audit(self, state: DefenseState, visual_description: str) -> Dict[str, Any]:
        system = (
            "You are the Visual Audit Agent (VAA) responsible for post-generation safety auditing "
            "from an objective image description. Judge only visible content in the description, "
            "while using upstream risk hypotheses to determine what to examine. Return strict JSON."
        )
        user = f"""UPSTREAM RISK HYPOTHESES:
{state.risk_hypotheses}

ORIGINAL PROMPT:
{state.original_prompt}

FINAL PROMPT SUBMITTED TO THE BLACK-BOX MODEL:
{state.current_prompt}

OBJECTIVE IMAGE DESCRIPTION:
{visual_description}

OUTPUT JSON:
{{
  "decision": "pass | block",
  "categories": ["sexual|violence|self-harm|hate|drugs|minors|exploitation"],
  "visible_evidence": ["evidence visible in the description"],
  "confidence": 0.0,
  "reason": "one concise English reason"
}}"""
        return self.llm.chat_json(system, user, max_tokens=600)


    def _visual_memory_score(self, state: DefenseState, visual_description: str) -> Dict[str, Any]:
        # Paper Sec. 3.7(c): compare the current visual description with the
        # retrieved historical visual cues and threshold the best similarity.
        if self.config.ablate_visual_failure_memory:
            return {
                "s_vis_mem": 0.0,
                "threshold": self.config.vaa_tau_vis_mem,
                "decision": "pass",
                "hits": [],
                "disabled": True,
            }
        if not self.config.failure_memory_enabled or not Path(self.config.failure_memory_path).exists():
            return {
                "s_vis_mem": 0.0,
                "threshold": self.config.vaa_tau_vis_mem,
                "decision": "pass",
                "hits": [],
            }
        try:
            from .failure_memory import FailureMemoryStore, cue_iou

            store = FailureMemoryStore(
                self.config.failure_memory_path,
                enabled=self.config.failure_memory_enabled,
                max_lessons=self.config.failure_memory_max_lessons,
            )
            lessons = store.relevant_lessons(state.current_prompt or state.original_prompt, visual_description)
            best = 0.0
            hits: List[str] = []
            categories: List[str] = []
            for lesson in lessons:
                cues = lesson.get("visual_cues", []) if isinstance(lesson, dict) else []
                score = cue_iou(visual_description, cues if isinstance(cues, list) else [])
                if score > best:
                    best = score
                    hits = [str(item) for item in cues if str(item).strip()][:12] if isinstance(cues, list) else []
                    categories = [str(item) for item in lesson.get("categories", []) if str(item).strip()] if isinstance(lesson, dict) else []
            return {
                "s_vis_mem": round(best, 6),
                "threshold": self.config.vaa_tau_vis_mem,
                "decision": "block" if best >= float(self.config.vaa_tau_vis_mem) else "pass",
                "hits": hits,
                "categories": categories,
            }
        except Exception as exc:
            state.errors.append(f"VAAgent visual memory failed: {type(exc).__name__}: {exc}")
            return {
                "s_vis_mem": 0.0,
                "threshold": self.config.vaa_tau_vis_mem,
                "decision": "pass",
                "hits": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
