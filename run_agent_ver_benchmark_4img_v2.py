from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_ROOT = SCRIPT_DIR.parent
FINAL_SCRIPTS = AGENT_ROOT.parent / "final" / "scripts"
BASE_RUNNER_PATH = AGENT_ROOT / "ADD" / "scripts" / "run_agent_ver_benchmark.py"

for path in (AGENT_ROOT, FINAL_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from four_image_utils import FIXED_SEEDS, aggregate_seed_judges, prompt_level_decision, representative_seed_result, state_has_prompt_judges


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("agent_ver_single_runner", BASE_RUNNER_PATH)

from agent_ver_2 import AgentVersionConfig, CoordinatorAgent  # noqa: E402
from agent_ver_2.failure_memory import append_failure_case, build_failure_case, update_lessons  # noqa: E402
from official_judges import I2POfficialJudges  # noqa: E402


METRIC_FIELDS = [
    "benchmark",
    "method",
    "row_index",
    "sample_id",
    "category",
    "source",
    "prompt_chars",
    "fixed_seed_count",
    "fixed_seeds",
    "final_decision",
    "block_stage",
    "iga_status",
    "ssa_status",
    "prompt_success",
    "success_stage",
    "success_reason",
    "pre_generation_blocked",
    "generated_seed_count",
    "blocked_seed_count",
    "harmful_seed_count",
    "q16_fail_count",
    "nudenet_fail_count",
    "allow_original_seed_count",
    "allow_sanitized_seed_count",
    "vaa_block_seed_count",
    "harmful_seeds",
    "row_seconds",
    "prompt_defense_seconds",
    "generation_total_seconds",
    "visual_description_total_seconds",
    "va_agent_total_seconds",
    "official_judges_total_seconds",
    "error",
    "state_path",
    "image_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent-ver with the 4-image worst-case protocol.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--method", default="V3-gpt-mini-sd14-4img")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sd-service-url", default="http://127.0.0.1:8010")
    parser.add_argument("--sd-model-path", type=Path, default=None)
    parser.add_argument("--vlm-service-url", default="http://127.0.0.1:8011")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--indices-file", type=Path, default=None)
    parser.add_argument("--gpt-config", type=Path, default=Path("config") / "gpt_config.json")
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--chat-completions", action="store_true")
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-ssa-on-pass", action="store_true")
    parser.add_argument("--sample-retries", type=int, default=BASE.DEFAULT_SAMPLE_RETRIES)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--disable-failure-memory", action="store_true")
    parser.add_argument("--failure-memory-path", type=Path, default=None)
    parser.add_argument("--failure-memory-read-only", action="store_true")
    parser.add_argument("--ig-report-only", action="store_true", help="Keep IG evidence but downgrade IG reject to SSA handoff.")
    parser.add_argument("--continue-after-text-block", action="store_true", help="For VAA stress tests, still generate/audit when text-side defense would block.")
    parser.add_argument("--generate-original-for-vaa", action="store_true", help="For VAA stress tests, generate the original prompt after recording IGA/SSA reports.")
    parser.add_argument("--ablate-conditioned-vlm", action="store_true")
    parser.add_argument("--ablate-visual-failure-memory", action="store_true")
    parser.add_argument("--ablate-casr", action="store_true")
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "images": out_dir / "images",
        "states": out_dir / "states",
        "tables": out_dir / "tables",
        "logs": out_dir / "logs",
        "summary": out_dir / "summary",
        "errors": out_dir / "errors",
        "failure_memory": out_dir / "failure_memory",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def get_failure_memory_path(args: argparse.Namespace, dirs: Dict[str, Path]) -> Path:
    return args.failure_memory_path or (dirs["failure_memory"] / "lessons.json")


def build_config(args: argparse.Namespace, failure_memory_path: Path) -> AgentVersionConfig:
    config = AgentVersionConfig(
        qwen=BASE.load_llm_config(args),
        run_ssa_on_pass=not args.no_ssa_on_pass,
    )
    config.generation.service_url = args.sd_service_url
    if args.sd_model_path is not None:
        config.generation.model_path = args.sd_model_path
    config.vision.service_url = args.vlm_service_url
    config.failure_memory_path = failure_memory_path
    config.failure_memory_enabled = not args.disable_failure_memory
    config.failure_memory_read_only = bool(args.failure_memory_read_only)
    config.ig_allow_block = not args.ig_report_only
    config.ablate_conditioned_vlm = bool(args.ablate_conditioned_vlm)
    config.ablate_visual_failure_memory = bool(args.ablate_visual_failure_memory)
    config.ablate_casr = bool(args.ablate_casr)
    return config


def record_failure_memory_for_prompt(
    representative: Dict[str, Any] | None,
    failure_memory_dir: Path,
    failure_memory_path: Path,
    state_path: Path,
    source_run: str,
    read_only: bool = False,
) -> None:
    # Paper Sec. 3.3 and Sec. 4.1: only externally judged unsafe generations
    # update persistent memory; read-only mode is used for controlled ablations.
    if read_only:
        return
    if not representative:
        return
    payload = copy.deepcopy(representative)
    payload["state_path"] = str(state_path)
    case = build_failure_case(payload, source_run=source_run)
    if not case:
        return
    append_failure_case(failure_memory_dir / "failed_cases.jsonl", case)
    update_lessons(failure_memory_path, case)


def build_metric(
    args: argparse.Namespace,
    row_index: int,
    row: Dict[str, str],
    payload: Dict[str, Any],
    row_seconds: float,
    state_path: Path,
    image_dir: Path,
    error: str = "",
) -> Dict[str, Any]:
    official = payload.get("i2p_official_judges") or {}
    timings = payload.get("timings", {}) if isinstance(payload.get("timings"), dict) else {}
    seed_results = payload.get("seed_results", []) if isinstance(payload.get("seed_results"), list) else []
    allow_original = sum(1 for item in seed_results if str(item.get("final_decision") or "") == "allow_original")
    allow_sanitized = sum(1 for item in seed_results if str(item.get("final_decision") or "") == "allow_sanitized")
    vaa_block = sum(
        1
        for item in seed_results
        if str(item.get("final_decision") or "") == "block" and str(item.get("block_stage") or "") == "va_agent"
    )
    return {
        "benchmark": args.benchmark,
        "method": args.method,
        "row_index": row_index,
        "sample_id": row.get("sample_id", ""),
        "category": row.get("category", ""),
        "source": row.get("source", ""),
        "prompt_chars": len(row.get("prompt", "")),
        "fixed_seed_count": len(FIXED_SEEDS),
        "fixed_seeds": ",".join(map(str, FIXED_SEEDS)),
        "final_decision": payload.get("final_decision", ""),
        "block_stage": payload.get("block_stage", ""),
        "iga_status": payload.get("iga_status", ""),
        "ssa_status": payload.get("ssa_status", ""),
        "prompt_success": official.get("benchmark_success", ""),
        "success_stage": official.get("success_stage", ""),
        "success_reason": official.get("success_reason", ""),
        "pre_generation_blocked": official.get("pre_generation_blocked", ""),
        "generated_seed_count": official.get("generated_seed_count", ""),
        "blocked_seed_count": official.get("blocked_seed_count", ""),
        "harmful_seed_count": official.get("harmful_seed_count", ""),
        "q16_fail_count": official.get("q16_fail_count", ""),
        "nudenet_fail_count": official.get("nudenet_fail_count", ""),
        "allow_original_seed_count": allow_original,
        "allow_sanitized_seed_count": allow_sanitized,
        "vaa_block_seed_count": vaa_block,
        "harmful_seeds": ",".join(map(str, official.get("harmful_seeds", []))),
        "row_seconds": round(row_seconds, 4),
        "prompt_defense_seconds": timings.get("prompt_defense", ""),
        "generation_total_seconds": timings.get("generation_total", ""),
        "visual_description_total_seconds": timings.get("visual_description_total", ""),
        "va_agent_total_seconds": timings.get("va_agent_total", ""),
        "official_judges_total_seconds": timings.get("official_judges_total", ""),
        "error": error,
        "state_path": str(state_path),
        "image_dir": str(image_dir),
    }


def run_one(
    row_index: int,
    row: Dict[str, str],
    args: argparse.Namespace,
    dirs: Dict[str, Path],
    judges: I2POfficialJudges,
) -> Dict[str, Any]:
    # Paper Sec. 4.1: evaluate each prompt with four fixed seeds under the
    # prompt-level worst-case rule, auditing every released image separately.
    # A prompt fails when any released image is classified as unsafe; otherwise
    # the prompt is considered successfully defended.
    sample_id = row.get("sample_id") or f"row_{row_index:05d}"
    prompt = row["prompt"]
    state_path = dirs["states"] / f"{sample_id}.json"
    image_dir = dirs["images"] / sample_id
    row_start = time.perf_counter()
    failure_memory_path = get_failure_memory_path(args, dirs)
    coordinator = CoordinatorAgent(build_config(args, failure_memory_path))
    try:
        prompt_start = time.perf_counter()
        prompt_state = coordinator.defend_prompt(prompt)
        prompt_seconds = round(time.perf_counter() - prompt_start, 4)

        payload = prompt_state.to_dict()
        payload["benchmark"] = args.benchmark
        payload["method"] = args.method
        payload["row_index"] = row_index
        payload["sample"] = dict(row)
        payload["sample_id"] = sample_id
        payload["fixed_seeds"] = list(FIXED_SEEDS)
        payload["experiment"] = {
            "protocol": "4img_worst_case",
            "sd_service_url": args.sd_service_url,
            "sd_model_path": str(args.sd_model_path) if args.sd_model_path else "",
            "vlm_service_url": args.vlm_service_url,
            "llm_model": coordinator.config.qwen.model,
            "failure_memory_path": str(failure_memory_path),
            "failure_memory_read_only": bool(args.failure_memory_read_only),
        }
        payload.setdefault("timings", {})["prompt_defense"] = prompt_seconds

        if payload.get("final_decision") == "block":
            if args.continue_after_text_block:
                payload["text_side_would_block"] = {
                    "final_decision": payload.get("final_decision"),
                    "block_stage": payload.get("block_stage"),
                    "rejection_reason": payload.get("rejection_reason", ""),
                }
                prompt_state.tool_outputs["text_side_would_block"] = payload["text_side_would_block"]
                prompt_state.final_decision = "needs_image_audit"
                prompt_state.block_stage = None
                prompt_state.rejection_reason = ""
                prompt_state.vaa_status = "pending"
                prompt_state.current_prompt = prompt_state.current_prompt or prompt_state.normalized_prompt or prompt_state.original_prompt
                if args.generate_original_for_vaa:
                    prompt_state.current_prompt = prompt_state.original_prompt
                    prompt_state.sanitized_prompt = ""
                    prompt_state.tool_outputs["vaa_stress_generation_prompt"] = "original_prompt"
                payload = prompt_state.to_dict()
                payload["benchmark"] = args.benchmark
                payload["method"] = args.method
                payload["row_index"] = row_index
                payload["sample"] = dict(row)
                payload["sample_id"] = sample_id
                payload["fixed_seeds"] = list(FIXED_SEEDS)
                payload["experiment"] = {
                    "protocol": "4img_worst_case",
                    "sd_service_url": args.sd_service_url,
                    "sd_model_path": str(args.sd_model_path) if args.sd_model_path else "",
                    "vlm_service_url": args.vlm_service_url,
                    "llm_model": coordinator.config.qwen.model,
                    "failure_memory_path": str(failure_memory_path),
                    "failure_memory_read_only": bool(args.failure_memory_read_only),
                    "continue_after_text_block": True,
                    "generate_original_for_vaa": bool(args.generate_original_for_vaa),
                }
                payload.setdefault("timings", {})["prompt_defense"] = prompt_seconds
            else:
                official = aggregate_seed_judges(
                    [],
                    pre_generation_blocked=True,
                    block_stage=str(payload.get("block_stage") or ""),
                    elapsed_seconds=0.0,
                )
                payload["seed_results"] = []
                payload["i2p_official_judges"] = official
                row_seconds = time.perf_counter() - row_start
                payload["timings"]["row_total"] = round(row_seconds, 4)
                BASE.write_json(state_path, payload)
                return {
                    "state": payload,
                    "metric": build_metric(args, row_index, row, payload, row_seconds, state_path, image_dir),
                }

        if args.generate_original_for_vaa and payload.get("final_decision") != "block":
            prompt_state.tool_outputs["vaa_stress_generation_prompt"] = {
                "mode": "original_prompt",
                "reported_iga_status": prompt_state.iga_status,
                "reported_ssa_status": prompt_state.ssa_status,
                "reported_sanitized_prompt": prompt_state.sanitized_prompt,
            }
            prompt_state.current_prompt = prompt_state.original_prompt
            prompt_state.sanitized_prompt = ""
            prompt_state.ssa_status = "report_only_original_generation"
            payload = prompt_state.to_dict()
            payload["benchmark"] = args.benchmark
            payload["method"] = args.method
            payload["row_index"] = row_index
            payload["sample"] = dict(row)
            payload["sample_id"] = sample_id
            payload["fixed_seeds"] = list(FIXED_SEEDS)
            payload["experiment"] = {
                "protocol": "4img_worst_case",
                "sd_service_url": args.sd_service_url,
                "sd_model_path": str(args.sd_model_path) if args.sd_model_path else "",
                "vlm_service_url": args.vlm_service_url,
                "llm_model": coordinator.config.qwen.model,
                "failure_memory_path": str(failure_memory_path),
                "failure_memory_read_only": bool(args.failure_memory_read_only),
                "continue_after_text_block": bool(args.continue_after_text_block),
                "generate_original_for_vaa": True,
            }
            payload.setdefault("timings", {})["prompt_defense"] = prompt_seconds

        seed_results: List[Dict[str, Any]] = []
        generation_total = 0.0
        visual_description_total = 0.0
        va_agent_total = 0.0
        official_total = 0.0

        for seed in FIXED_SEEDS:
            seed_state = copy.deepcopy(prompt_state)
            image_path = image_dir / f"seed_{seed}.png"

            generation_start = time.perf_counter()
            generation = coordinator.generation_tool.generate(seed_state.current_prompt, image_path, seed=seed)
            generation_total += time.perf_counter() - generation_start
            seed_state.image_path = generation["image_path"]
            seed_state.sd_status = "generated"
            seed_state.tool_outputs["generation_tool"] = generation
            seed_state.timings["generation_tool"] = generation.get("seconds", "")
            seed_state.add_route("blackbox_t2i:generated")

            audit_start = time.perf_counter()
            coordinator.audit_image(seed_state, image_path)
            audit_elapsed = time.perf_counter() - audit_start
            visual_description_total += float(seed_state.timings.get("visual_description_tool", 0.0) or 0.0)
            va_agent_total += float(seed_state.timings.get("va_agent", 0.0) or 0.0)

            seed_payload = seed_state.to_dict()
            seed_payload["benchmark"] = args.benchmark
            seed_payload["method"] = args.method
            seed_payload["row_index"] = row_index
            seed_payload["sample_id"] = sample_id
            seed_payload["sample"] = dict(row)
            seed_payload["seed"] = seed
            seed_payload["i2p_index"] = row_index
            seed_payload.setdefault("timings", {})["seed_total"] = round(
                float(seed_payload["timings"].get("generation_tool", 0.0) or 0.0) + audit_elapsed,
                4,
            )

            judge_start = time.perf_counter()
            BASE.attach_official_judges(seed_payload, judges)
            official_total += time.perf_counter() - judge_start
            seed_results.append(seed_payload)

        final_decision, block_stage = prompt_level_decision(seed_results)
        payload["final_decision"] = final_decision
        payload["block_stage"] = block_stage
        payload["seed_results"] = seed_results
        representative = representative_seed_result(seed_results)
        payload["seed"] = (representative or {}).get("seed")
        payload["i2p_index"] = row_index
        payload["image_path"] = str((representative or {}).get("image_path") or "")
        payload["tool_outputs"] = dict(payload.get("tool_outputs") or {})
        if representative:
            representative_tools = representative.get("tool_outputs")
            if isinstance(representative_tools, dict) and representative_tools.get("visual_description_tool"):
                payload["tool_outputs"]["visual_description_tool"] = representative_tools["visual_description_tool"]

        aggregate_seconds = time.perf_counter() - row_start
        official = aggregate_seed_judges(
            seed_results,
            pre_generation_blocked=False,
            block_stage=block_stage,
            elapsed_seconds=official_total,
        )
        payload["i2p_official_judges"] = official
        payload["timings"]["generation_total"] = round(generation_total, 4)
        payload["timings"]["visual_description_total"] = round(visual_description_total, 4)
        payload["timings"]["va_agent_total"] = round(va_agent_total, 4)
        payload["timings"]["official_judges_total"] = round(official_total, 4)
        payload["timings"]["row_total"] = round(aggregate_seconds, 4)
        record_failure_memory_for_prompt(
            representative if official.get("benchmark_success") is False else None,
            dirs["failure_memory"],
            failure_memory_path,
            state_path,
            f"{args.benchmark}/{args.method}",
            read_only=bool(args.failure_memory_read_only),
        )
        BASE.write_json(state_path, payload)
        error_path = dirs["errors"] / f"{sample_id}.json"
        if error_path.exists():
            error_path.unlink()
        return {
            "state": payload,
            "metric": build_metric(args, row_index, row, payload, aggregate_seconds, state_path, image_dir),
        }
    except Exception as exc:
        row_seconds = time.perf_counter() - row_start
        error = f"{type(exc).__name__}: {exc}"
        error_payload = {
            "benchmark": args.benchmark,
            "method": args.method,
            "row_index": row_index,
            "sample_id": sample_id,
            "sample": dict(row),
            "error": error,
            "traceback": traceback.format_exc(),
            "timings": {"row_total": round(row_seconds, 4)},
        }
        error_path = dirs["errors"] / f"{sample_id}.json"
        BASE.write_json(error_path, error_payload)
        state = {
            "final_decision": "error",
            "iga_status": "",
            "ssa_status": "",
            "seed_results": [],
            "timings": {"row_total": round(row_seconds, 4)},
            "i2p_official_judges": {
                "protocol": "4img_worst_case",
                "benchmark_success": "",
                "success_stage": "",
                "success_reason": "",
                "pre_generation_blocked": "",
                "generated_seed_count": "",
                "blocked_seed_count": "",
                "harmful_seed_count": "",
                "q16_fail_count": "",
                "nudenet_fail_count": "",
                "harmful_seeds": [],
            },
        }
        return {
            "state": error_payload,
            "metric": build_metric(args, row_index, row, state, row_seconds, error_path, image_dir, error=error),
        }
    finally:
        coordinator.close()


def main() -> int:
    args = parse_args()
    dirs = ensure_dirs(args.out_dir)
    failure_memory_path = get_failure_memory_path(args, dirs)
    failure_memory_path.parent.mkdir(parents=True, exist_ok=True)
    rows = BASE.read_rows(args.dataset)
    indices = list(BASE.selected_indices(len(rows), args.start_index, args.limit, args.indices_file))
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "benchmark": args.benchmark,
        "method": args.method,
        "dataset_rows": len(rows),
        "selected_rows": len(indices),
        "out_dir": str(args.out_dir),
        "sd_service_url": args.sd_service_url,
        "sd_model_path": str(args.sd_model_path) if args.sd_model_path else "",
        "vlm_service_url": args.vlm_service_url,
        "resume": not args.no_resume,
        "run_ssa_on_pass": not args.no_ssa_on_pass,
        "sample_retries": args.sample_retries,
        "fixed_seeds": list(FIXED_SEEDS),
        "failure_memory_path": str(failure_memory_path),
        "failure_memory_enabled": not args.disable_failure_memory,
        "failure_memory_read_only": bool(args.failure_memory_read_only),
        "ig_report_only": bool(args.ig_report_only),
        "continue_after_text_block": bool(args.continue_after_text_block),
        "generate_original_for_vaa": bool(args.generate_original_for_vaa),
        "ablate_conditioned_vlm": bool(args.ablate_conditioned_vlm),
        "ablate_visual_failure_memory": bool(args.ablate_visual_failure_memory),
        "ablate_casr": bool(args.ablate_casr),
    }
    BASE.write_json(args.out_dir / "manifest.json", manifest)

    metrics_path = dirs["tables"] / "metrics.csv"
    records_path = dirs["tables"] / "records.jsonl"
    progress_path = args.out_dir / "progress.json"
    processed = skipped = errors = 0
    print(
        f"[agent-ver-4img] benchmark={args.benchmark} method={args.method} rows={len(rows)} selected={len(indices)} out={args.out_dir}",
        flush=True,
    )
    judges = I2POfficialJudges()
    try:
        for row_index in indices:
            row = rows[row_index]
            sample_id = row.get("sample_id") or f"row_{row_index:05d}"
            state_path = dirs["states"] / f"{sample_id}.json"
            if not args.no_resume and state_path.exists():
                payload = BASE.read_json(state_path)
                if state_has_prompt_judges(payload):
                    skipped += 1
                    print(f"[agent-ver-4img] skip sample_id={sample_id}", flush=True)
                    continue

            print(f"[agent-ver-4img] start sample_id={sample_id}", flush=True)
            result = None
            for attempt in range(1, max(1, args.sample_retries) + 1):
                if attempt > 1:
                    print(
                        f"[agent-ver-4img] retry sample_id={sample_id} attempt={attempt}/{args.sample_retries}",
                        flush=True,
                    )
                result = run_one(row_index, row, args, dirs, judges)
                if not result["metric"].get("error"):
                    break
                if attempt < max(1, args.sample_retries):
                    time.sleep(min(5.0 * attempt, 20.0))
            assert result is not None

            metric = result["metric"]
            BASE.append_jsonl(records_path, result["state"])
            BASE.append_metric(metrics_path, metric)
            processed += 1
            if metric.get("error"):
                errors += 1
            if processed % max(1, args.flush_every) == 0:
                BASE.write_json(
                    progress_path,
                    {
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "processed_this_run": processed,
                        "skipped_existing": skipped,
                        "errors_this_run": errors,
                        "last_metric": metric,
                    },
                )
            print(
                "[agent-ver-4img] done sample_id={sid} success={success} harmful={harmful} blocked={blocked} seconds={sec}".format(
                    sid=sample_id,
                    success=metric.get("prompt_success"),
                    harmful=metric.get("harmful_seed_count"),
                    blocked=metric.get("blocked_seed_count"),
                    sec=metric.get("row_seconds"),
                ),
                flush=True,
            )
    finally:
        judges.close()

    BASE.write_json(
        progress_path,
        {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "processed_this_run": processed,
            "skipped_existing": skipped,
            "errors_this_run": errors,
            "finished": True,
        },
    )
    print(f"[agent-ver-4img] finished processed={processed} skipped={skipped} errors={errors}", flush=True)
    return 0


if __name__ == "__main__":
    BASE.METRIC_FIELDS = METRIC_FIELDS
    raise SystemExit(main())
