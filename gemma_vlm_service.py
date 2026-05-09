from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import traceback
from typing import Any, Callable, Dict, Optional
import urllib.error
import urllib.request


SERVICE_NAME = "llama_vlm_service_v1"
DEFAULT_MODEL_ID = "ggml-org/gemma-4-E4B-it-GGUF"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8777
DEFAULT_LLAMA_URL = "http://127.0.0.1:8778"
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_SEC = 120.0
GENERIC_MOTION_PATTERN = re.compile(
    r"\b(walk|run|jog|turn|rotate|pivot|wave|point|look|scan|crouch|duck|squat|"
    r"jump|hop|dance|idle|shift|fidget|stand|stop|stay|step)\b"
)
GO_TO_ACTION_NAMES = {
    "go_to",
    "goto",
    "walk_to",
    "move_to",
    "navigate_to",
    "go to",
    "walk to",
    "move to",
    "navigate to",
}

HttpPoster = Callable[[str, Dict[str, Any], float], Dict[str, Any]]
HttpGetter = Callable[[str, float], Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent llama.cpp-backed VLM decision adapter."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Adapter bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Adapter TCP port.")
    parser.add_argument(
        "--llama-url",
        default=DEFAULT_LLAMA_URL,
        help="Base URL for llama-server, without a trailing path.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Model id reported by the adapter and sent to llama-server.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    return parser.parse_args()


@dataclass
class LlamaCppVlmBackend:
    model_id: str = DEFAULT_MODEL_ID
    llama_url: str = DEFAULT_LLAMA_URL
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    http_post: Optional[HttpPoster] = None
    http_get: Optional[HttpGetter] = None

    def __post_init__(self) -> None:
        self.llama_url = self.llama_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model_id": self.model_id,
            "llama_url": self.llama_url,
            "upstream_ok": False,
        }
        try:
            self._get_json("/v1/models")
        except Exception as exc:  # noqa: BLE001
            payload["upstream_error"] = str(exc)
        else:
            payload["upstream_ok"] = True
            payload["upstream_error"] = ""
        return payload

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_payload = build_llama_chat_request(
            payload=payload,
            model_id=self.model_id,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        response = self._post_json("/v1/chat/completions", request_payload)
        raw_text = extract_llama_chat_content(response).strip()
        return {
            "ok": True,
            "service": SERVICE_NAME,
            "model_id": self.model_id,
            "raw_text": raw_text,
            "action": normalize_action(extract_json_object(raw_text)),
        }

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.llama_url + path
        if self.http_post is not None:
            return self.http_post(url, payload, self.timeout_sec)
        return post_json(url, payload, self.timeout_sec)

    def _get_json(self, path: str) -> Dict[str, Any]:
        url = self.llama_url + path
        if self.http_get is not None:
            return self.http_get(url, self.timeout_sec)
        return get_json(url, self.timeout_sec)


class GemmaVlmBackend(LlamaCppVlmBackend):
    """Backward-compatible class name for older local imports/tests."""


def build_llama_chat_request(
    payload: Dict[str, Any],
    model_id: str,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": build_llama_message_content(payload),
            }
        ],
        "max_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "stream": False,
        "response_format": {"type": "json_object"},
    }


def build_llama_message_content(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    controller_prompt = str(payload.get("controller_prompt", "")).strip() or default_controller_prompt()
    content: list[Dict[str, Any]] = [
        {
            "type": "text",
            "text": controller_prompt + "\n\n" + build_context_text(payload),
        }
    ]
    for frame in payload.get("frames", []):
        data_url = frame_data_url(frame)
        if data_url:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )
    content.append(
        {
            "type": "text",
            "text": "Choose the next action from the newest frame context. Return JSON only.",
        }
    )
    return content


def build_context_text(payload: Dict[str, Any]) -> str:
    context = {
        "schema_version": payload.get("schema_version"),
        "recent_actions": payload.get("recent_actions", []),
        "recent_decisions": payload.get("recent_decisions", []),
        "current_pose": payload.get("current_pose"),
        "current_position_m": payload.get("current_position_m"),
        "current_floor_position_m": payload.get("current_floor_position_m"),
        "pose_error": payload.get("pose_error"),
        "default_goal_location": payload.get("default_goal_location"),
        "activity_hint": payload.get("activity_hint"),
        "world_memory": payload.get("world_memory"),
    }
    return "Decision context JSON:\n" + json.dumps(context, ensure_ascii=True, sort_keys=True)


def frame_data_url(frame: Any) -> str:
    if not isinstance(frame, dict):
        return ""
    data_url = str(frame.get("data_url", "")).strip()
    if data_url:
        return data_url
    encoded = str(frame.get("image_base64", "")).strip()
    if not encoded:
        return ""
    mime_type = str(frame.get("mime_type", "image/png")).strip() or "image/png"
    return "data:{mime};base64,{payload}".format(mime=mime_type, payload=encoded)


def extract_llama_chat_content(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        content = response.get("content")
        return content if isinstance(content, str) else ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") in {None, "text"}
            )
    text = first.get("text")
    return text if isinstance(text, str) else ""


class GemmaVlmRequestHandler(BaseHTTPRequestHandler):
    backend: Any = None

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(404, {"ok": False, "error": "not_found"})
            return
        health = {}
        if hasattr(self.backend, "health"):
            health_result = self.backend.health()
            if isinstance(health_result, dict):
                health = health_result
        self._write_json(200, {"ok": True, "service": SERVICE_NAME, **health})

    def do_POST(self) -> None:
        if self.path != "/decide":
            self._write_json(404, {"ok": False, "error": "not_found"})
            return
        try:
            payload = self._read_json_body()
            response = self.backend.decide(payload)
            if not isinstance(response, dict):
                response = {"ok": False, "error": "Backend returned a non-object response."}
            self._write_json(200, response)
        except Exception as exc:  # noqa: BLE001
            print(traceback.format_exc(), flush=True)
            self._write_json(500, {"ok": False, "service": SERVICE_NAME, "error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        payload = json.loads(body or "{}")
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        encoded = (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def create_http_server(host: str, port: int, backend: Any) -> ThreadingHTTPServer:
    class _Handler(GemmaVlmRequestHandler):
        pass

    _Handler.backend = backend
    return ThreadingHTTPServer((host, int(port)), _Handler)


def post_json(url: str, payload: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError("llama.cpp request failed: {error}".format(error=exc)) from exc
    return decode_json_body(body)


def get_json(url: str, timeout_sec: float) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError("llama.cpp health request failed: {error}".format(error=exc)) from exc
    return decode_json_body(body)


def decode_json_body(body: str) -> Dict[str, Any]:
    try:
        decoded = json.loads(body or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("llama.cpp returned invalid JSON: {body}".format(body=body)) from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("llama.cpp returned a non-object JSON response.")
    return decoded


def default_controller_prompt() -> str:
    return (
        "You control a VRChat avatar by selecting text-to-motion actions from live "
        "screen images. Be action-forward: if the scene contains a nearby avatar, "
        "object, path, obstacle, social cue, or the avatar has been idle, choose "
        "generate_motion with a short physical response. Use noop only when the "
        "scene is unreadable, a recent action already covers the moment, or no "
        "safe/relevant movement can be inferred. Return exactly one JSON object "
        "with this schema: "
        '{"action":"noop","reason":"..."} or '
        '{"action":"generate_motion","prompt":"short motion prompt",'
        '"motion_length":4.0,"reset_session":false} or '
        '{"action":"reset_session","reason":"..."}. '
        "DART receives only the prompt text and cannot see the image, so never "
        "mention visible targets or scene references such as nearest avatar, "
        "that person, the object, the mirror, the door, or towards it. Translate "
        "what you see into one generic action primitive, such as walk forward, "
        "walk backward, turn left, turn right, wave, point forward, look around, "
        "step back, crouch, jump, dance, idle shift, or stand still. Do not choose "
        "OSC, avatar parameters, chat, or any non-motion action."
    )


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(str(text)):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(str(text)[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_action(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"action": "noop", "reason": "invalid_json"}
    action = str(payload.get("action", "")).strip()
    action_key = action.lower()
    if action_key == "noop":
        return {
            "action": "noop",
            "reason": str(payload.get("reason", "")),
        }
    if action_key == "generate_motion":
        return normalize_generate_motion(payload)
    if is_go_to_action(action):
        prompt = str(payload.get("prompt", "")).strip() or "walk"
        return normalize_generate_motion({**payload, "action": "generate_motion", "prompt": prompt})
    if is_motion_shorthand_action(action):
        prompt = str(payload.get("prompt", "")).strip() or action
        return normalize_generate_motion({**payload, "action": "generate_motion", "prompt": prompt})
    if action_key == "reset_session":
        return {
            "action": "reset_session",
            "reason": str(payload.get("reason", "")),
        }
    return {"action": "noop", "reason": "unsupported_action: {action}".format(action=action)}


def normalize_generate_motion(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        return {"action": "noop", "reason": "missing_prompt"}
    result: Dict[str, Any] = {
        "action": "generate_motion",
        "prompt": prompt,
        "reset_session": bool(payload.get("reset_session", False)),
    }
    motion_length = coerce_positive_float(payload.get("motion_length"))
    if motion_length is not None:
        result["motion_length"] = motion_length
    goal_location = candidate_goal_location(payload)
    if goal_location is not None:
        result["goal_location"] = goal_location
    angle = candidate_angle(payload)
    if angle is not None:
        result[angle[0]] = angle[1]
    reason = str(payload.get("reason", ""))
    if reason:
        result["reason"] = reason
    return result


def is_motion_shorthand_action(action: str) -> bool:
    normalized = " ".join(str(action).strip().lower().split())
    if not normalized or len(normalized.split()) > 8:
        return False
    return bool(GENERIC_MOTION_PATTERN.search(normalized))


def is_go_to_action(action: str) -> bool:
    normalized = " ".join(str(action).strip().lower().replace("_", " ").split())
    compact = normalized.replace(" ", "_")
    return normalized in GO_TO_ACTION_NAMES or compact in GO_TO_ACTION_NAMES


def candidate_goal_location(payload: Dict[str, Any]) -> Any:
    for key in (
        "goal_location",
        "target_location",
        "target_position",
        "destination",
        "coordinates",
    ):
        if key in payload:
            return payload.get(key)
    return None


def candidate_angle(payload: Dict[str, Any]) -> Optional[tuple[str, Any]]:
    for key in (
        "angle_degrees",
        "turn_degrees",
        "rotation_degrees",
        "angle",
        "degrees",
        "goal_angle",
        "angle_radians",
        "turn_radians",
        "rotation_radians",
        "radians",
    ):
        if key in payload:
            return key, payload.get(key)
    return None


def coerce_positive_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def main() -> int:
    args = parse_args()
    backend = LlamaCppVlmBackend(
        model_id=args.model_id,
        llama_url=args.llama_url,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    server = create_http_server(args.host, int(args.port), backend)
    print(
        "llama.cpp VLM adapter ready on {host}:{port}; upstream {url}; model {model}".format(
            host=args.host,
            port=args.port,
            url=args.llama_url.rstrip("/"),
            model=args.model_id,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
