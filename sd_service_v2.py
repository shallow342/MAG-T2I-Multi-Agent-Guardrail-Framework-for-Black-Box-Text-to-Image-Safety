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

from agent_ver_2.generation_tool import TextToImageGenerationTool  # noqa: E402
from agent_ver_2.schema import GenerationConfig  # noqa: E402


class SDService:
    # Paper Sec. 3.8: resident Stable Diffusion service is the black-box
    # text-in/image-out target interface used by the benchmark runner.
    def __init__(self, tool: TextToImageGenerationTool):
        self.tool = tool
        self._lock = threading.Lock()

    def handle_generate(self, payload: dict) -> dict:
        with self._lock:
            result = self.tool.generate(
                str(payload["prompt"]),
                str(payload["output_path"]),
                seed=int(payload.get("seed", 0)),
            )
        return {"ok": True, "result": result}


def make_handler(service: SDService):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._send({"ok": True, "service": "agent_ver_2_sd"})
                return
            self._send({"ok": False, "error": "not found"}, status=404)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/generate":
                    self._send(service.handle_generate(payload))
                else:
                    self._send({"ok": False, "error": "not found"}, status=404)
            except Exception as exc:
                self._send({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

        def log_message(self, fmt, *args):
            print(f"[agent-ver-2-sd] {self.address_string()} - {fmt % args}", flush=True)

        def _send(self, payload: dict, status: int = 200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Resident SD service for agent_ver_2.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8310)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    config = GenerationConfig(service_url="")
    config.model_path = args.model_path
    config.device = args.device
    tool = TextToImageGenerationTool(config)
    if args.warmup:
        print("[agent-ver-2-sd] warming up Stable Diffusion pipeline...", flush=True)
        tool._load_pipeline()
        print("[agent-ver-2-sd] warmup done", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(SDService(tool)))
    print(f"[agent-ver-2-sd] listening on http://{args.host}:{args.port} model_path={config.model_path}", flush=True)
    try:
        server.serve_forever()
    finally:
        tool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
