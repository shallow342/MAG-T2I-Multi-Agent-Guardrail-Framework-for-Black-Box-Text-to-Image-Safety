from __future__ import annotations

import time
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, Sequence

from .schema import VisionConfig


def conditioned_vlm_prompt(
    base_prompt: str,
    risk_hypotheses: Sequence[str] | None = None,
    *,
    ablate_conditioned_vlm: bool = False,
) -> str:
    # Paper Sec. 3.7(a): condition visual description on upstream hypotheses H
    # while retaining the constraint to report only visible evidence.
    # The additional questions direct the VLM to examine categories identified
    # by upstream agents.
    if ablate_conditioned_vlm or not risk_hypotheses:
        return base_prompt
    risks = sorted({str(item).strip() for item in risk_hypotheses if str(item).strip()})
    if not risks:
        return base_prompt
    focus = ", ".join(risks[:8])
    return (
        f"{base_prompt}\n\n"
        "Conditioning for this audit: upstream text-side agents raised these risk hypotheses: "
        f"{focus}. Still describe only visible content. Pay particular attention to visible cues that confirm "
        "or refute those hypotheses, and write '- none' when no such visible cue appears."
    )


class QwenVLVisualDescriptionTool:
    """Self-contained Qwen2.5-VL image description tool for VAAgent."""

    def __init__(self, config: VisionConfig):
        self.config = config
        self._model = None
        self._processor = None

    def describe(
        self,
        image_path: str | Path,
        risk_hypotheses: Sequence[str] | None = None,
        *,
        ablate_conditioned_vlm: bool = False,
        prompt_override: str | None = None,
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            raise RuntimeError("VisionConfig.enabled is false.")
        start = time.perf_counter()
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        prompt = prompt_override or conditioned_vlm_prompt(
            self.config.prompt,
            risk_hypotheses,
            ablate_conditioned_vlm=ablate_conditioned_vlm,
        )
        if self.config.service_url.strip():
            return self._describe_with_service(image_path, prompt=prompt, risk_hypotheses=risk_hypotheses)
        model, processor = self._load_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return {
            "image_path": str(image_path),
            "description": output_text,
            "model_path": str(self.config.model_path),
            "backend": "native_transformers",
            "conditioned": prompt != self.config.prompt,
            "risk_hypotheses": list(risk_hypotheses or []),
            "seconds": round(time.perf_counter() - start, 4),
        }

    def close(self) -> None:
        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
            self._model = None
            self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_model(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            from transformers import AutoProcessor, QwenVLForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "Qwen2.5-VL requires a recent transformers build. "
                "Install the required packages and make the local model checkpoint available."
            ) from exc

        kwargs = {
            "dtype": self.config.dtype,
            "local_files_only": True,
        }
        if self.config.device_map.strip():
            kwargs["device_map"] = self.config.device_map
        self._model = QwenVLForConditionalGeneration.from_pretrained(
            str(self.config.model_path),
            **kwargs,
        )
        if not self.config.device_map.strip():
            self._model = self._model.to(self.config.device)
        self._processor = AutoProcessor.from_pretrained(
            str(self.config.model_path),
            local_files_only=True,
        )
        return self._model, self._processor

    def _describe_with_service(
        self,
        image_path: Path,
        *,
        prompt: str,
        risk_hypotheses: Sequence[str] | None,
    ) -> Dict[str, Any]:
        payload = {
            "image_path": str(image_path),
            "prompt": prompt,
            "risk_hypotheses": list(risk_hypotheses or []),
        }
        request = urllib.request.Request(
            self.config.service_url.rstrip("/") + "/describe",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not data.get("ok", False):
            raise RuntimeError(str(data.get("error", "VLM service description failed")))
        return data["result"]
