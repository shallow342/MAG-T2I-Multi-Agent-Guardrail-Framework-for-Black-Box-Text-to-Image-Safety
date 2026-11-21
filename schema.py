from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


AgentName = Literal["coordinator", "ig_agent", "ss_agent", "va_agent", "blackbox_t2i"]
FinalDecision = Literal["block", "allow_original", "allow_sanitized", "needs_image_audit"]


@dataclass
class QwenLLMConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "qwen3-14b-awq"
    timeout: float = 45.0
    max_tokens: int = 900
    temperature: float = 0.0
    strip_thinking: bool = True


@dataclass
class AgentVersionConfig:
    # Paper Sec. 3.2/3.4/3.6.2/3.7: runtime switches and thresholds for the
    # shared-state coordinator, CASR scoring, conditional VLM, and ablations.
    qwen: QwenLLMConfig = field(default_factory=QwenLLMConfig)
    generation: "GenerationConfig" = field(default_factory=lambda: GenerationConfig())
    vision: "VisionConfig" = field(default_factory=lambda: VisionConfig())
    # Keep defaults portable; deployment-specific paths belong in runner config.
    failure_memory_path: Path = Path("failure_memory") / "lessons.json"
    failure_memory_enabled: bool = True
    failure_memory_read_only: bool = False
    failure_memory_max_lessons: int = 8
    run_ssa_on_pass: bool = True
    ig_allow_block: bool = True
    use_ig_llm_triage: bool = True
    use_va_vlm_description: bool = True
    use_ss_failure_memory: bool = True
    block_on_failed_high_risk_sanitize: bool = True
    high_risk_reject_threshold: int = 4
    # CASR ablation switches
    ablate_casr: bool = False
    casr_alpha: float = 0.4
    casr_beta: float = 0.4
    casr_gamma: float = 0.2
    casr_tau_safe: float = 0.1
    casr_tau_risky: float = 0.35
    # VAA ablation switches
    ablate_conditioned_vlm: bool = False
    ablate_visual_failure_memory: bool = False
    vaa_tau_vis_mem: float = 0.25


@dataclass
class GenerationConfig:
    enabled: bool = True
    service_url: str = ""
    model_path: Path = Path("models") / "stable-diffusion-2-1-base"
    device: str = "cuda"
    torch_dtype: str = "float16"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    disable_safety_checker: bool = True


@dataclass
class VisionConfig:
    enabled: bool = True
    service_url: str = ""
    model_path: Path = Path("models") / "qwen-vl"
    device: str = "cuda"
    device_map: str = ""
    dtype: str = "auto"
    max_new_tokens: int = 320
    do_sample: bool = False
    use_subprocess_fallback: bool = True
    subprocess_conda_env: str = "qwen"
    subprocess_timeout: float = 900.0
    subprocess_cuda_visible_devices: str = "0"
    # Optional deployment overrides; empty strings use the current environment.
    subprocess_ld_preload: str = ""
    subprocess_conda_init: str = ""
    prompt: str = (
        "You are the Visual Description Tool for a text-to-image safety defense system. "
        "Describe only visible content. Do not infer hidden intent or guess the prompt. "
        "Return exactly this format in English:\n"
        "Scene: <1-2 concise sentences about visible scene>\n"
        "RiskEvidence:\n"
        "- <one visible safety-relevant cue per bullet>\n"
        "If none are visible, write:\n"
        "RiskEvidence:\n"
        "- none\n"
        "Report only visible evidence relevant to sexual content, violence, self-harm, hate, drugs, harm to minors, "
        "exploitation, or adversarial text. Do not infer hidden intent or invent risk."
    )


@dataclass
class AgentEvidence:
    # Paper Sec. 3.2: one structured evidence item written to the shared
    # blackboard; its category is also propagated as a risk hypothesis.
    source: AgentName | str
    category: str
    severity: Literal["info", "low", "medium", "high"]
    confidence: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "category": self.category,
            "severity": self.severity,
            "confidence": self.confidence,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass
class DefenseState:
    # Paper Sec. 3.2: shared request state (prompt states, stage decisions,
    # routing, hypotheses, evidence, and tool outputs) visible to all agents.
    # The shared blackboard carries structured evidence and risk hypotheses
    # between the specialized agents.
    original_prompt: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    normalized_prompt: str = ""
    current_prompt: str = ""
    sanitized_prompt: str = ""
    image_path: str = ""
    iga_status: str = "pending"
    ssa_status: str = "pending"
    sd_status: str = "pending"
    vaa_status: str = "pending"
    final_decision: Optional[FinalDecision] = None
    block_stage: Optional[str] = None
    rejection_reason: str = ""
    route: List[str] = field(default_factory=list)
    risk_hypotheses: List[str] = field(default_factory=list)
    evidence: List[AgentEvidence] = field(default_factory=list)
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    agent_messages: List[Dict[str, Any]] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_route(self, item: str) -> None:
        self.route.append(item)

    def add_evidence(self, evidence: AgentEvidence) -> None:
        self.evidence.append(evidence)
        if evidence.category and evidence.category not in self.risk_hypotheses:
            self.risk_hypotheses.append(evidence.category)

    def add_message(self, agent: AgentName | str, message: str, metadata: Dict[str, Any] | None = None) -> None:
        self.agent_messages.append(
            {"agent": agent, "message": message, "metadata": dict(metadata or {})}
        )

    def risk_summary(self) -> str:
        if not self.evidence:
            return "no upstream evidence"
        parts = []
        for item in self.evidence[-12:]:
            parts.append(
                f"{item.source}:{item.category}:{item.severity}:{round(item.confidence, 3)}:{item.text}"
            )
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "original_prompt": self.original_prompt,
            "normalized_prompt": self.normalized_prompt,
            "current_prompt": self.current_prompt,
            "sanitized_prompt": self.sanitized_prompt,
            "image_path": self.image_path,
            "iga_status": self.iga_status,
            "ssa_status": self.ssa_status,
            "sd_status": self.sd_status,
            "vaa_status": self.vaa_status,
            "final_decision": self.final_decision,
            "block_stage": self.block_stage,
            "rejection_reason": self.rejection_reason,
            "route": list(self.route),
            "risk_hypotheses": list(self.risk_hypotheses),
            "evidence": [item.to_dict() for item in self.evidence],
            "tool_outputs": self.tool_outputs,
            "agent_messages": list(self.agent_messages),
            "timings": dict(self.timings),
            "errors": list(self.errors),
            "created_at": self.created_at,
        }
