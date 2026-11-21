from __future__ import annotations

import time
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict

from .schema import GenerationConfig


class TextToImageGenerationTool:
    """Self-contained local Stable Diffusion generation tool.

    This intentionally does not import the legacy VAA generator. It only uses
    diffusers directly when generation is requested.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._pipe = None

    def generate(self, prompt: str, output_path: str | Path, *, seed: int = 0) -> Dict[str, Any]:
        # Paper Sec. 3.8: text-in/image-out black-box generation interface;
        # this wrapper supports either a local Diffusers pipeline or HTTP SD service.
        # MAG-T2I submits prompts and receives generated images without accessing
        # target T2I weights, gradients, intermediate representations, or process.
        if not self.config.enabled:
            raise RuntimeError("GenerationConfig.enabled is false.")
        if self.config.service_url.strip():
            return self._generate_with_service(prompt, output_path, seed=seed)
        start = time.perf_counter()
        pipe = self._load_pipeline()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import torch

        generator = torch.Generator(device=self._generator_device()).manual_seed(seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=self.config.negative_prompt or None,
            width=self.config.width,
            height=self.config.height,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
        ).images[0]
        image.save(output_path)
        return {
            "image_path": str(output_path),
            "seed": seed,
            "prompt": prompt,
            "model_path": str(self.config.model_path),
            "width": self.config.width,
            "height": self.config.height,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "seconds": round(time.perf_counter() - start, 4),
        }

    def close(self) -> None:
        if self._pipe is not None:
            try:
                self._pipe.to("cpu")
            except Exception:
                pass
            self._pipe = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except Exception as exc:
            raise RuntimeError(
                "Stable Diffusion generation requires torch and diffusers in the current environment. "
                "Use the defense environment or install compatible packages."
            ) from exc

        dtype = self._resolve_dtype(torch)
        pipe = StableDiffusionPipeline.from_pretrained(
            str(self.config.model_path),
            torch_dtype=dtype,
            local_files_only=True,
            safety_checker=None if self.config.disable_safety_checker else None,
            requires_safety_checker=False,
        )
        if self.config.disable_safety_checker and hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        pipe = pipe.to(self.config.device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        self._pipe = pipe
        return self._pipe

    def _generator_device(self) -> str:
        if self.config.device.startswith("cuda"):
            return self.config.device
        return "cpu"

    def _resolve_dtype(self, torch):
        if self.config.torch_dtype == "float16":
            return torch.float16
        if self.config.torch_dtype == "bfloat16":
            return torch.bfloat16
        if self.config.torch_dtype == "float32":
            return torch.float32
        return None

    def _generate_with_service(self, prompt: str, output_path: str | Path, *, seed: int) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "output_path": str(output_path),
            "seed": seed,
        }
        request = urllib.request.Request(
            self.config.service_url.rstrip("/") + "/generate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not data.get("ok", False):
            raise RuntimeError(str(data.get("error", "SD service generation failed")))
        return data["result"]
