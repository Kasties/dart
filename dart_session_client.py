import argparse
import json
import socket
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a generation request to a running DART session service.")
    parser.add_argument("--host", default="127.0.0.1", help="Prompt service host.")
    parser.add_argument("--port", type=int, default=8765, help="Prompt service TCP port.")
    parser.add_argument("--task", default="generate_motion", help="Task name for protocol extensibility.")
    parser.add_argument("--prompt", required=True, help="Text prompt.")
    parser.add_argument("--frame-count", type=int, required=True, help="Frame count to generate.")
    parser.add_argument("--guidance-param", type=float, default=5.0, help="Classifier-free guidance parameter.")
    parser.add_argument("--output-dir", required=True, help="Remote output directory for results.")
    parser.add_argument("--job-id", default="", help="Optional caller job id.")
    parser.add_argument("--seed", type=int, help="Optional random seed.")
    parser.add_argument("--reset-session", action="store_true", help="Reset the remote DART session before generating.")
    parser.add_argument("--timeout-sec", type=float, default=900.0, help="Socket timeout in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request: Dict[str, Any] = {
        "task": args.task,
        "prompt": args.prompt,
        "frame_count": int(args.frame_count),
        "guidance_param": float(args.guidance_param),
        "output_dir": args.output_dir,
        "job_id": args.job_id,
        "reset_session": bool(args.reset_session),
    }
    if args.seed is not None:
        request["seed"] = int(args.seed)

    response = send_request(args.host, args.port, request, timeout_sec=args.timeout_sec)
    print(json.dumps(response, ensure_ascii=True))
    if not response.get("ok", False):
        return 1
    return 0


def send_request(host: str, port: int, request: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
    with socket.create_connection((host, port), timeout=timeout_sec) as sock:
        sock.settimeout(timeout_sec)
        payload = (json.dumps(request, ensure_ascii=True) + "\n").encode("utf-8")
        sock.sendall(payload)
        buffer = bytearray()
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buffer.extend(chunk)
            if b"\n" in chunk:
                break
    if not buffer:
        return {"ok": False, "error": "No response from DART session service."}
    line = buffer.decode("utf-8", errors="replace").splitlines()[0]
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"ok": False, "error": "Invalid JSON response from DART session service.", "raw": line}


if __name__ == "__main__":
    raise SystemExit(main())
