from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from agent_ver_2.agents import SSAgent  # noqa: E402
from agent_ver_2.failure_memory import FailureMemoryStore  # noqa: E402
from agent_ver_2.llm_client import QwenLLMClient  # noqa: E402
from agent_ver_2.schema import AgentVersionConfig, DefenseState, QwenLLMConfig  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_llm_config(path: Path) -> QwenLLMConfig:
    payload = read_json(path)
    base_url = str(payload.get("openai_base_url") or "http://127.0.0.1:8000/v1").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return QwenLLMConfig(
        enabled=True,
        base_url=base_url,
        api_key=str(payload.get("openai_api_key") or "EMPTY"),
        model=str(payload.get("openai_model") or "gpt-5.4-mini"),
        timeout=float(payload.get("openai_timeout", 90.0)),
        max_tokens=int(payload.get("openai_max_output_tokens", 1200)),
        temperature=float(payload.get("temperature", 0.0)),
    )


def prompt_from_row(row: dict[str, str]) -> str:
    for key in ("prompt", "original_prompt", "text", "adv_prompt"):
        if row.get(key):
            return str(row[key])
    return next((str(value) for value in row.values() if value), "")


def iter_rows(path: Path, start_index: int, limit: int):
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index < start_index:
                continue
            if limit and row_index >= start_index + limit:
                break
            yield row_index, row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-side SSA/CASR diagnosis only.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpt-config", type=Path, default=Path("config") / "gpt_config.json")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--failure-memory-path", type=Path, default=None)
    parser.add_argument("--disable-failure-memory", action="store_true")
    return parser.parse_args()


def main() -> int:
    # Paper Sec. 3.6 and supplementary diagnosis protocol: isolate SSA/CASR
    # decisions and export per-row states plus aggregate metrics.
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    states_dir = args.out_dir / "states"
    tables_dir = args.out_dir / "tables"
    states_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    config = AgentVersionConfig(qwen=load_llm_config(args.gpt_config))
    config.generation.enabled = False
    config.vision.enabled = False
    if args.failure_memory_path is not None:
        config.failure_memory_path = args.failure_memory_path
    if args.disable_failure_memory:
        config.failure_memory_enabled = False
        config.use_ss_failure_memory = False
    memory = FailureMemoryStore(
        config.failure_memory_path,
        enabled=config.failure_memory_enabled,
        max_lessons=config.failure_memory_max_lessons,
    )
    agent = SSAgent(QwenLLMClient(config.qwen), config)

    fieldnames = [
        "row_index",
        "ssa_status",
        "final_decision",
        "risk_score",
        "categories",
        "should_rewrite",
        "sanitized_prompt",
        "light_gate_applied",
        "route",
        "seconds",
        "state_path",
    ]
    counts = {"processed": 0, "pass_original": 0, "sanitize": 0, "block": 0, "errors": 0}
    metrics_path = tables_dir / "ssa_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as metrics:
        writer = csv.DictWriter(metrics, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, row in iter_rows(args.dataset, args.start_index, args.limit):
            prompt = prompt_from_row(row)
            state = DefenseState(original_prompt=prompt)
            state.request_id = f"row_{row_index:05d}"
            state.current_prompt = prompt
            state.normalized_prompt = prompt
            state.iga_status = "skipped"
            state.tool_outputs["failure_memory_context"] = {
                "lessons": memory.relevant_lessons(prompt) if config.failure_memory_enabled else [],
            }
            start = time.perf_counter()
            try:
                agent.run(state)
            except Exception as exc:
                state.errors.append(f"SSA diagnosis failed: {type(exc).__name__}: {exc}")
                counts["errors"] += 1
            seconds = round(time.perf_counter() - start, 4)
            state.timings["ssa_diagnosis_total"] = seconds
            payload = state.to_dict()
            payload["row_index"] = row_index
            payload["sample"] = row
            state_path = states_dir / f"row_{row_index:05d}.json"
            write_json(state_path, payload)
            analysis = state.tool_outputs.get("ss_agent_analysis", {})
            light_gate = state.tool_outputs.get("ss_agent_light_rewrite_gate", {})
            counts["processed"] += 1
            if state.final_decision == "block":
                counts["block"] += 1
            elif state.ssa_status == "sanitize":
                counts["sanitize"] += 1
            elif state.ssa_status == "pass_original":
                counts["pass_original"] += 1
            writer.writerow(
                {
                    "row_index": row_index,
                    "ssa_status": state.ssa_status,
                    "final_decision": state.final_decision or "",
                    "risk_score": analysis.get("risk_score", ""),
                    "categories": ",".join(analysis.get("categories", []) if isinstance(analysis.get("categories"), list) else []),
                    "should_rewrite": analysis.get("should_rewrite", ""),
                    "sanitized_prompt": state.sanitized_prompt,
                    "light_gate_applied": bool(light_gate.get("applied")) if isinstance(light_gate, dict) else "",
                    "route": " > ".join(state.route),
                    "seconds": seconds,
                    "state_path": str(state_path),
                }
            )

    write_json(
        args.out_dir / "ssa_summary.json",
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": str(args.dataset),
            "start_index": args.start_index,
            "limit": args.limit,
            **counts,
        },
    )
    print(f"[ssa-diagnosis] counts={counts} metrics={metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
