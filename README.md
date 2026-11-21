# MAG-T2I

MAG-T2I is a multimodal safety defense pipeline for text-to-image
generation. It checks a prompt before generation, audits the generated image
after generation, and records structured evidence so that every decision can
be inspected.

The main path is:

```text
prompt
  -> IGA: fast input triage
  -> SSA: semantic analysis and CASR sanitization
  -> Stable Diffusion: image generation
  -> VLM: visible-content description
  -> VAA: visual safety audit
  -> block / allow_original / allow_sanitized
```

The system is not only a keyword filter. IGA detects explicit and suspicious
input cues, SSA handles implicit risk and can rewrite a prompt in two CASR
stages, and VAA checks whether the released image is safe even when the text
side passed. Failure memory stores structured lessons from previous failures
and makes later prompt and image audits more conservative.

## Source files

All paths below are relative to the repository root.

| File | Role |
| --- | --- |
| `__init__.py` | Exposes the public package objects, including `CoordinatorAgent`, `AgentVersionConfig`, `DefenseState`, and `AgentEvidence`. |
| `schema.py` | Defines configuration dataclasses and the shared `DefenseState` / `AgentEvidence` records. |
| `agents.py` | Implements IGA, SSA, and VAA, including lexical checks, semantic analysis, CASR L1/L2 rewriting, verification, and visual-risk signals. |
| `coordinator.py` | Orchestrates the complete text-defense, generation, visual-description, and visual-audit pipeline. |
| `failure_memory.py` | Builds failure cases, aggregates lessons, and retrieves relevant lessons for prompts and visual descriptions. |
| `generation_tool.py` | Provides local Diffusers generation and HTTP-client mode for a resident Stable Diffusion service. |
| `vision_tool.py` | Provides local VLM description, HTTP-client mode, conditioned visual prompting. |
| `llm_client.py` | Implements the OpenAI-compatible chat-completions client used by the agents. |
| `json_utils.py` | Extracts JSON objects and removes model thinking blocks from LLM responses. |
| `run_ssa_diagnosis.py` | Runs prompt-side SSA/CASR diagnosis without image generation or visual auditing. |
| `run_agent_ver_benchmark_4img_v2.py` | Runs the four-image worst-case benchmark protocol, writes per-sample states and metrics, and updates failure memory when enabled. |
| `sd_service_v2.py` | Hosts a resident Stable Diffusion HTTP service with `/health` and `/generate`. |
| `vlm_service_v2.py` | Hosts a resident VLM HTTP service with `/health` and `/describe`. |
| `requirement.txt` | Lists Python packages required by the local model backends. |
| `LICENSE` | MIT license for the source code in this directory. |
| `Dataset` | The safety benchmarks used in the experiments. |

## Core state model

`DefenseState` in `schema.py` is the shared blackboard. Every agent reads from
it and appends its decision, evidence, route, timing, and tool output.

The fields most useful when debugging a sample are:

```text
original_prompt       original user input
normalized_prompt     normalized text used by the text-side agents
current_prompt        prompt actually sent to the generator
sanitized_prompt      CASR output, when SSA rewrites the prompt
iga_status             IGA stage result
ssa_status             SSA stage result
sd_status              generation result
vaa_status             visual-audit result
final_decision         end-to-end decision
block_stage            stage responsible for a block
rejection_reason       human-readable block reason
risk_hypotheses        risk categories propagated to later agents
evidence               structured AgentEvidence entries
route                  ordered execution trace
tool_outputs           machine-readable intermediate results
timings                per-stage elapsed times
errors                 non-fatal errors captured during execution
```

`AgentVersionConfig` controls the LLM, generation, vision, failure-memory,
threshold, and ablation settings. Important switches include
`run_ssa_on_pass`, `failure_memory_read_only`, `ablate_casr`,
`ablate_conditioned_vlm`, and `ablate_visual_failure_memory`.

## How the pipeline works

### IGA

`IGAgent` performs fast input triage with explicit-risk, malformed/attack-
structure, and suggestive-cue tools. It can pass the prompt, hand it to SSA,
or block it immediately when configured to allow direct blocking.

### SSA and CASR

`SSAgent` analyzes implicit semantics and decides whether to keep the original
prompt, sanitize it, or block it. CASR decomposes a prompt into slots, scores
their lexical/semantic/memory risk, and tries:

1. L1: local replacement that preserves safe content where possible.
2. L2: abstraction of only the slots that failed L1 verification.

Each candidate is checked by the LLM verifier and a lexical risk verifier. If both stages fail, the prompt is blocked and a lesson can be written
to failure memory.

### Generation

`TextToImageGenerationTool` uses `state.current_prompt`. It can load a local
Diffusers pipeline or call a resident SD service. The seed is passed to a
PyTorch generator and the result is recorded in `state.image_path` and
`tool_outputs["generation_tool"]`.

### VLM and VAA

`QwenVLVisualDescriptionTool` converts an image into a structured visible-
content description. The prompt can be conditioned on upstream
`risk_hypotheses`, but the VLM is instructed to report only visible evidence.

`VAAgent` combines three channels:

```text
vrl      visible-risk lexicon checks
vis_mem  visual failure-memory matches
llm      LLM audit of visible evidence and upstream hypotheses
```

The signals are stored in
`tool_outputs["va_agent_all_channel_signals"]`. A blocking channel is recorded
in `tool_outputs["va_agent_block_channel"]`; the combined audit is stored in
`tool_outputs["va_agent_audit"]`.

### Failure memory

Failure memory is a JSON lesson store, not model training. A failure case can
contain prompt cues, visual cues, judge results, risk categories, and
recommended actions. `FailureMemoryStore` retrieves matching lessons before
SSA and VAA run. For a frozen evaluation split, use
`failure_memory_read_only=True` or the corresponding runner flag so that the
evaluation does not update the memory.

## Installation and configuration

The package uses Python's standard library for orchestration and HTTP clients.
Local model backends additionally require PyTorch, Transformers, Diffusers,
Accelerate, and Safetensors in versions compatible with the selected models.

Install the listed dependencies from the repository root with:

```bash
python -m pip install -r agent_ver_2/requirement.txt
```

For GPU inference, install the PyTorch wheel matching the target CUDA runtime
before running the services. The model weights themselves are not installed by
this file.

From the repository root, install the environment required by the local
backends according to the model versions used in the experiment. Model weights,
datasets, generated images, and API credentials are not included in this
source release and must be obtained under their own terms.

The command-line runners accept an OpenAI-compatible endpoint. A minimal
runtime config can be stored at `config/gpt_config.json`:

```json
{
  "openai_base_url": "http://127.0.0.1:8000/v1",
  "openai_api_key": "EMPTY",
  "openai_model": "qwen3-14b-awq",
  "openai_timeout": 90,
  "openai_max_output_tokens": 1200,
  "temperature": 0.0
}
```

The diagnosis runner accepts CSV rows with a `prompt` column. It also accepts
`original_prompt`, `text`, or `adv_prompt` when `prompt` is absent.

## Running the code

Run source-only checks from the repository root:

```bash
python -m compileall -q agent_ver_2
python agent_ver_2/run_ssa_diagnosis.py --help
python agent_ver_2/sd_service_v2.py --help
python agent_ver_2/vlm_service_v2.py --help
```

### Python API

For text-side inspection, call `defend_prompt` directly:

```python
from agent_ver_2 import AgentVersionConfig, CoordinatorAgent

coordinator = CoordinatorAgent(AgentVersionConfig())
try:
    state = coordinator.defend_prompt("A neutral landscape at sunrise.")
    print(state.final_decision)
    print(state.to_dict())
finally:
    coordinator.close()
```

The coordinator also exposes `audit_image(state, image_path)` for an existing
image, `defend_with_visual_description(prompt, visual_description)` for an
offline VAA audit, and `defend_full(prompt, output_image_path, seed=42)` for
the complete text, generation, and image-audit path. The latter requires the
configured local models or the resident services.

### Prompt-side SSA/CASR diagnosis

This mode does not load image-generation or VLM weights:

```bash
python agent_ver_2/run_ssa_diagnosis.py \
  --dataset data/prompts.csv \
  --out-dir runs/ssa \
  --gpt-config config/gpt_config.json \
  --start-index 0 \
  --limit 20
```

It writes `states/row_<index>.json`, `tables/ssa_metrics.csv`, and
`ssa_summary.json` below `runs/ssa`.

### Resident Stable Diffusion service

```bash
python agent_ver_2/sd_service_v2.py \
  --model-path models/stable-diffusion-2-1-base \
  --host 127.0.0.1 \
  --port 8010 \
  --device cuda \
  --warmup
```

Check it with:

```bash
curl http://127.0.0.1:8010/health
```

### Resident VLM service

```bash
python agent_ver_2/vlm_service_v2.py \
  --model-path models/qwen-vl \
  --host 127.0.0.1 \
  --port 8011 \
  --device cuda \
  --warmup
```

Check it with:

```bash
curl http://127.0.0.1:8011/health
```

### Four-image benchmark

The benchmark runner evaluates each prompt with four fixed seeds, audits each
generated image independently, and applies the prompt-level worst-case rule:
one externally judged harmful seed makes the prompt fail. Run it from the
repository root after the complete benchmark environment and its external
judges have been configured:

```bash
python agent_ver_2/run_agent_ver_benchmark_4img_v2.py \
  --dataset data/i2p.csv \
  --benchmark I2P \
  --out-dir runs/i2p \
  --gpt-config config/gpt_config.json \
  --sd-service-url http://127.0.0.1:8010 \
  --vlm-service-url http://127.0.0.1:8011 \
  --start-index 0 \
  --limit 20
```

Useful runner switches include:

```text
--no-resume                    reprocess existing states
--disable-llm                  disable LLM calls where supported
--disable-failure-memory       disable lesson retrieval and writes
--failure-memory-path PATH     choose the lesson JSON file
--failure-memory-read-only     freeze lesson writes for evaluation
--no-ssa-on-pass               skip SSA after an IGA pass
--ablate-casr                  disable CASR
--ablate-conditioned-vlm       remove upstream hypotheses from VLM prompts
--ablate-visual-failure-memory disable VAA visual-memory signals
--continue-after-text-block    generate images for VAA stress tests
--generate-original-for-vaa    use the original prompt in VAA stress tests
```

## Output layout

For a runner output such as `runs/i2p`, the important files are:

```text
runs/i2p/
  manifest.json                 run configuration and fixed seeds
  progress.json                 progress and final status
  images/<sample_id>/           generated images, one file per seed
  states/<sample_id>.json       complete prompt-level state
  tables/metrics.csv            one structured metric row per sample
  tables/records.jsonl          complete state records, one per line
  errors/<sample_id>.json       traceback and metadata for failed rows
  failure_memory/               lessons and failed cases when enabled
```

The four-image protocol uses fixed seeds and records them in the manifest and
state payload. Runtime artifacts should remain outside the source files.

## Reading one sample's state JSON

Suppose a state file is at `runs/i2p/states/sample_0001.json`. Inspect it with
the standard library:

```python
import json
from pathlib import Path

path = Path("runs/i2p/states/sample_0001.json")
state = json.loads(path.read_text(encoding="utf-8"))

print(state["final_decision"])
print(state.get("block_stage"))
print(state.get("rejection_reason", ""))
```

Read the fields in this order:

1. `final_decision`: `block`, `needs_image_audit`, `allow_original`, or
   `allow_sanitized`.
2. `block_stage` and `rejection_reason`: where and why a block happened.
3. `iga_status`, `ssa_status`, `sd_status`, and `vaa_status`: the result of
   each stage.
4. `route`: the actual execution path, such as SSA sanitization followed by
   generation and VAA.
5. `risk_hypotheses`: risk categories propagated between agents.
6. `current_prompt` and `sanitized_prompt`: the prompt used for generation and
   the CASR output, if any.
7. `tool_outputs`: detailed intermediate evidence. Common keys are
   `ig_agent_tools`, `ss_agent_analysis`, `ss_agent_casr_decomposition`,
   `ss_agent_verifier`, `ss_agent_casr_l2`, `generation_tool`,
   `visual_description_tool`, `va_agent_all_channel_signals`,
   `va_agent_block_channel`, `va_agent_audit`, and
   `failure_memory_context`.
8. `seed_results`: in the four-image runner, one nested state per fixed seed;
   inspect each seed before relying on the aggregate result.
9. `i2p_official_judges`: aggregate external-judge counts and the final
   prompt-level success decision, when the benchmark protocol is used.
10. `timings` and `errors`: performance data and non-fatal failure details.

For a quick summary of every seed in a benchmark record:

```python
for result in state.get("seed_results", []):
    print(
        result.get("seed"),
        result.get("final_decision"),
        result.get("block_stage"),
        result.get("image_path", ""),
    )
```

`ssa_status="sanitize"` is not the same as final safety: it only means that
the text-side rewrite passed. The image still needs VLM/VAA auditing and any
external benchmark judges.

## Decisions and development notes

```text
block              the system rejects the prompt or image
needs_image_audit  text-side checks passed; image audit is still pending
allow_original     original prompt was used and the image audit passed
allow_sanitized    sanitized prompt was used and the image audit passed
```

Failure memory is a JSON record, not parameter training. In a frozen
evaluation, enable read-only mode. Keep the names of `DefenseState` fields and
the main `tool_outputs` keys stable because analysis scripts consume them.

To modify the system, start with `agents.py` for IGA/SSA/VAA behavior,
`failure_memory.py` for lesson construction and retrieval,
`coordinator.py` for routing, and the runner for benchmark orchestration.

## License

The source code in this directory is released under the MIT License. See
[`LICENSE`](LICENSE). Model weights, datasets, generated images, external
judges, and hosted APIs are governed by their own licenses or terms.
