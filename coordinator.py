from __future__ import annotations

import time
from pathlib import Path

from .agents import IGAgent, SSAgent, VAAgent
from .failure_memory import FailureMemoryStore
from .generation_tool import TextToImageGenerationTool
from .llm_client import QwenLLMClient
from .schema import AgentEvidence, AgentVersionConfig, DefenseState
from .vision_tool import QwenVLVisualDescriptionTool


class CoordinatorAgent:
    """Coordinator with a shared blackboard state.

    This implementation is intentionally self-contained inside agent-ver. It
    does not import legacy IGA/SSA/VAA modules.
    """

    def __init__(self, config: AgentVersionConfig | None = None):
        self.config = config or AgentVersionConfig()
        self.llm = QwenLLMClient(self.config.qwen)
        self.ig_agent = IGAgent(self.llm, self.config)
        self.ss_agent = SSAgent(self.llm, self.config)
        self.va_agent = VAAgent(self.llm, self.config)
        self.generation_tool = TextToImageGenerationTool(self.config.generation)
        self.visual_description_tool = QwenVLVisualDescriptionTool(self.config.vision)
        self.failure_memory = FailureMemoryStore(
            self.config.failure_memory_path,
            enabled=self.config.failure_memory_enabled,
            max_lessons=self.config.failure_memory_max_lessons,
        )

    def defend_prompt(self, prompt: str) -> DefenseState:
        # Paper Sec. 3.1 and Sec. 3.4 (CSA): initialize the blackboard, retrieve
        # cross-request lessons, then route every non-blocked prompt to SSA.
        # CSA prioritizes intent-preserving safe rewriting and blocks only when
        # a clear risk is present or cannot be eliminated through rewriting.
        state = DefenseState(original_prompt=prompt)
        total_start = time.perf_counter()
        state.add_message(
            "coordinator",
            "Initialize the shared state and start the text-side collaborative defense between IG Agent and SS Agent.",
            {"qwen_available": self.llm.available, "legacy_imports": False},
        )
        self._attach_failure_memory(state)

        self.ig_agent.run(state)
        if state.iga_status == "reject":
            state.final_decision = "block"
            state.block_stage = "ig_agent"
            state.rejection_reason = self._joined_ig_reasons(state)
            state.ssa_status = "skipped"
            state.vaa_status = "not_applicable"
            state.timings["coordinator_total"] = round(time.perf_counter() - total_start, 4)
            return state

        if state.iga_status == "handoff_to_ssa" or self.config.run_ssa_on_pass:
            self.ss_agent.run(state)
        else:
            state.ssa_status = "skipped"
            state.add_route("ss_agent:skipped")

        if state.final_decision == "block":
            state.vaa_status = "not_applicable"
        else:
            state.final_decision = "needs_image_audit"
            state.current_prompt = state.current_prompt or state.normalized_prompt or state.original_prompt

        state.timings["coordinator_total"] = round(time.perf_counter() - total_start, 4)
        return state

    def audit_visual_description(self, state: DefenseState, visual_description: str) -> DefenseState:
        # Paper Sec. 3.7: run the post-generation visual audit before release.
        if state.final_decision == "block":
            return state
        self.va_agent.run(state, visual_description)
        if state.final_decision != "block":
            state.final_decision = "allow_sanitized" if state.ssa_status == "sanitize" else "allow_original"
        return state

    def defend_with_visual_description(self, prompt: str, visual_description: str) -> DefenseState:
        state = self.defend_prompt(prompt)
        return self.audit_visual_description(state, visual_description)

    def audit_image(self, state: DefenseState, image_path: str | Path) -> DefenseState:
        # Paper Sec. 3.7(a): obtain a conditional visual description from VLM,
        # conditioned on upstream risk hypotheses stored in the blackboard.
        if state.final_decision == "block":
            return state
        start = time.perf_counter()
        description_result = self.visual_description_tool.describe(
            image_path,
            risk_hypotheses=state.risk_hypotheses,
            ablate_conditioned_vlm=self.config.ablate_conditioned_vlm,
        )
        state.image_path = str(image_path)
        state.tool_outputs["visual_description_tool"] = description_result
        state.timings["visual_description_tool"] = round(time.perf_counter() - start, 4)
        state.add_route("visual_description_tool:describe")
        return self.audit_visual_description(state, description_result["description"])

    def defend_image(self, prompt: str, image_path: str | Path) -> DefenseState:
        state = self.defend_prompt(prompt)
        return self.audit_image(state, image_path)

    def defend_full(self, prompt: str, output_image_path: str | Path, *, seed: int = 0) -> DefenseState:
        # Paper Sec. 3.1/3.8: black-box text-in/image-out inference followed by
        # VAA; the target T2I model is never accessed internally.
        state = self.defend_prompt(prompt)
        if state.final_decision == "block":
            return state
        start = time.perf_counter()
        generation_result = self.generation_tool.generate(state.current_prompt, output_image_path, seed=seed)
        state.image_path = generation_result["image_path"]
        state.sd_status = "generated"
        state.tool_outputs["generation_tool"] = generation_result
        state.timings["generation_tool"] = round(time.perf_counter() - start, 4)
        state.add_route("blackbox_t2i:generated")
        self.generation_tool.close()
        return self.audit_image(state, output_image_path)

    def close(self) -> None:
        self.va_agent.close()
        self.generation_tool.close()
        self.visual_description_tool.close()

    @staticmethod
    def _joined_ig_reasons(state: DefenseState) -> str:
        findings = state.tool_outputs.get("ig_agent_tools", [])
        reasons = []
        if isinstance(findings, list):
            for item in findings:
                if isinstance(item, dict) and item.get("decision") == "reject":
                    reasons.extend(str(reason) for reason in item.get("reasons", []))
        return "; ".join(reasons) or "blocked by IG Agent"

    def _attach_failure_memory(self, state: DefenseState) -> None:
        # Paper Sec. 3.3/3.4: pre-generation retrieval makes lessons available
        # to SSA through the shared blackboard.
        lessons = self.failure_memory.relevant_lessons(state.original_prompt)
        state.tool_outputs["failure_memory_context"] = {
            "path": str(self.config.failure_memory_path),
            "matched_count": len(lessons),
            "lessons": lessons,
        }
        if not lessons:
            return
        state.add_route("failure_memory:matched")
        for lesson in lessons[:3]:
            categories = lesson.get("categories", [])
            category = str(categories[0]) if isinstance(categories, list) and categories else "jailbreak"
            cues = lesson.get("matched_prompt_cues", [])
            actions = lesson.get("recommended_actions", [])
            text_parts = []
            if cues:
                text_parts.append("matched cues: " + ", ".join(map(str, cues[:8])))
            if actions:
                text_parts.append("lesson: " + str(actions[0]))
            state.add_evidence(
                AgentEvidence(
                    source="failure_memory",
                    category=category,
                    severity="high",
                    confidence=0.85,
                    text="; ".join(text_parts) or "matched previous official-judge failure pattern",
                    metadata={
                        "memory_key": lesson.get("key", ""),
                        "count": lesson.get("count", 0),
                        "example_indices": lesson.get("example_indices", []),
                        "max_q16_probability_inappropriate": lesson.get(
                            "max_q16_probability_inappropriate", 0
                        ),
                    },
                )
            )
