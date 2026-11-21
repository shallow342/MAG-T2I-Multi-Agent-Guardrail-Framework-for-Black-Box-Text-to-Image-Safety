from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from agent_ver_2.schema import VisionConfig  # noqa: E402
from agent_ver_2.vision_tool import QwenVLVisualDescriptionTool  # noqa: E402


class VLMService:
    # Paper Sec. 3.7(a) and Sec. 3.8: resident VLM endpoint exposes the
    # image-description operation used by VAA through a text/JSON interface.
    def __init__(self, tool: QwenVLVisualDescriptionTool):
        self.tool = tool
        self._lock = threading.Lock()

    def handle_describe(self, payload: dict) -> dict:
        with self._lock:
            result = self.tool.describe(
                str(payload["image_path"]),
                risk_hypotheses=payload.get("risk_hypotheses") or None,
                ablate_conditioned_vlm=bool(payload.get("ablate_conditioned_vlm", False)),
                prompt_override=payload.get("prompt") or None,
            )
        return {"ok": True, "result": result}


def make_handler(service: VLMService):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._send({"ok": True, "service": "agent_ver_2_vlm"})
                return
            self._send({"ok": False, "error": "not found"}, status=404)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/describe":
                    self._send(service.handle_describe(payload))
                else:
                    self._send({"ok": False, "error": "not found"}, status=404)
            except Exception as exc:
                self._send({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

        def log_message(self, fmt, *args):
            print(f"[agent-ver-2-vlm] {self.address_string()} - {fmt % args}", flush=True)

        def _send(self, payload: dict, status: int = 200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Resident Qwen2.5-VL service for agent_ver_2.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8311)
    parser.add_argument("--model-path", type=Path, default=VisionConfig().model_path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    config = VisionConfig(service_url="", use_subprocess_fallback=False)
    config.model_path = args.model_path
    config.device = args.device
    tool = QwenVLVisualDescriptionTool(config)
    if args.warmup:
        print("[agent-ver-2-vlm] warming up Qwen2.5-VL...", flush=True)
        tool._load_model()
        print("[agent-ver-2-vlm] warmup done", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(VLMService(tool)))
    print(f"[agent-ver-2-vlm] listening on http://{args.host}:{args.port} model_path={config.model_path}", flush=True)
    try:
        server.serve_forever()
    finally:
        tool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
