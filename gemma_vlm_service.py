from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import traceback
from typing import Any, Callable, Dict, Optional, Sequence
import urllib.error
import urllib.request


SERVICE_NAME = "llama_vlm_service_v1"
SUPPORTED_BACKENDS = ("llama", "openrouter")
DEFAULT_MODEL_ID = "ggml-org/gemma-4-E4B-it-GGUF"
DEFAULT_OPENROUTER_MODEL_ID = "google/gemini-2.5-flash"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8777
DEFAULT_LLAMA_URL = "http://127.0.0.1:8778"
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_SEC = 120.0
DEFAULT_RESPONSE_FORMAT = False
UPSTREAM_IMAGE_COUNT = 1
WORLD_MEMORY_CELL_COUNT = 4
WORLD_MEMORY_WALK_ATTEMPT_COUNT = 2
WORLD_MEMORY_FRONTIER_GOAL_COUNT = 2
CONTEXT_KEYS = (
    "activity_hint",
    "current_floor_position_m",
    "default_goal_location",
    "pose_error",
    "recent_actions",
    "world_memory",
)
GENERIC_MOTION_PATTERN = re.compile(
    r"\b(walk|run|jog|turn|rotate|pivot|wave|point|look|scan|crouch|duck|squat|"
    r"jump|hop|dance|idle|shift|fidget|stand|stop|stay|step)\b"
)
SCENE_REFERENCE_PATTERN = re.compile(
    r"\b(nearest|nearby|closest|avatar|player|person|character|object|door|doorway|"
    r"mirror|path|obstacle|wall|chair|thing|target|them|him|her|it)\b|"
    r"\b(towards?|to|at|near|beside|behind|through|into|onto|from)\b"
)
MOTION_TAG_PATTERN = re.compile(r"\{\s*motion\s*:\s*([^{}]*)\}", re.IGNORECASE)
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
        description="Persistent VLM decision adapter."
    )
    parser.add_argument(
        "--backend",
        choices=SUPPORTED_BACKENDS,
        default="llama",
        help="VLM backend to use. openrouter calls the OpenRouter API instead of llama-server.",
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
        default=None,
        help="Model id sent to the selected backend. Defaults depend on --backend.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--openrouter-api-key",
        default="",
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--openrouter-url",
        default="",
        help="OpenRouter API base URL or chat completions endpoint.",
    )
    parser.add_argument(
        "--openrouter-referer",
        default="",
        help="Optional HTTP-Referer header for OpenRouter rankings.",
    )
    parser.add_argument(
        "--openrouter-title",
        default="",
        help="Optional X-OpenRouter-Title header.",
    )
    parser.add_argument(
        "--openrouter-provider",
        default="",
        help="Optional comma-separated OpenRouter provider slug order, for example alibaba.",
    )
    parser.add_argument(
        "--openrouter-allow-fallbacks",
        dest="openrouter_allow_fallbacks",
        action="store_true",
        default=None,
        help="Allow OpenRouter to fall back to other providers after the requested provider order.",
    )
    parser.add_argument(
        "--openrouter-no-provider-fallbacks",
        dest="openrouter_allow_fallbacks",
        action="store_false",
        help="Require the requested OpenRouter provider order without falling back to other providers.",
    )
    parser.add_argument(
        "--openrouter-response-format",
        dest="openrouter_response_format",
        action="store_true",
        default=None,
        help="Request OpenRouter JSON response_format. Disabled by default for free-text motion tags.",
    )
    parser.add_argument(
        "--openrouter-no-response-format",
        dest="openrouter_response_format",
        action="store_false",
        help="Do not send response_format to OpenRouter. This is the default free-text motion-tag mode.",
    )
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
        action = normalize_model_response(raw_text)
        return {
            "ok": True,
            "service": SERVICE_NAME,
            "model_id": self.model_id,
            "raw_text": raw_text,
            "speech_text": str(action.get("speech_text", "")),
            "action": action,
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


@dataclass
class OpenRouterVlmBackend:
    model_id: str = DEFAULT_OPENROUTER_MODEL_ID
    api_key: str = ""
    openrouter_url: str = DEFAULT_OPENROUTER_URL
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    referer: str = ""
    title: str = "VRCAI"
    provider_order: tuple[str, ...] = ()
    allow_fallbacks: Optional[bool] = None
    response_format: bool = DEFAULT_RESPONSE_FORMAT
    http_post: Optional[HttpPoster] = None

    def __post_init__(self) -> None:
        self.openrouter_url = self.openrouter_url.rstrip("/")
        self.provider_order = tuple(provider for provider in self.provider_order if provider)
        if not self.api_key and self.http_post is None:
            raise ValueError("OpenRouter backend requires OPENROUTER_API_KEY or --openrouter-api-key.")

    def health(self) -> Dict[str, Any]:
        return {
            "backend": "openrouter",
            "model_id": self.model_id,
            "openrouter_url": self._chat_completions_url(),
            "provider_order": list(self.provider_order),
            "allow_fallbacks": self.allow_fallbacks,
            "response_format": bool(self.response_format),
            "api_key_configured": bool(self.api_key),
        }

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_payload = build_llama_chat_request(
            payload=payload,
            model_id=self.model_id,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            include_response_format=self.response_format,
        )
        provider_config = build_openrouter_provider_config(
            provider_order=self.provider_order,
            allow_fallbacks=self.allow_fallbacks,
        )
        if provider_config:
            request_payload["provider"] = provider_config
        response = self._post_json(request_payload)
        raw_text = extract_llama_chat_content(response).strip()
        action = normalize_model_response(raw_text)
        return {
            "ok": True,
            "service": SERVICE_NAME,
            "backend": "openrouter",
            "model_id": self.model_id,
            "provider": provider_config,
            "raw_text": raw_text,
            "speech_text": str(action.get("speech_text", "")),
            "action": action,
        }

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._chat_completions_url()
        if self.http_post is not None:
            return self.http_post(url, payload, self.timeout_sec)
        headers = self._headers()
        return post_json_with_headers(
            url=url,
            payload=payload,
            timeout_sec=self.timeout_sec,
            headers=headers,
            error_context="OpenRouter",
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Authorization": "Bearer {key}".format(key=self.api_key)}
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-OpenRouter-Title"] = self.title
        return headers

    def _chat_completions_url(self) -> str:
        if self.openrouter_url.endswith("/chat/completions"):
            return self.openrouter_url
        return self.openrouter_url + "/chat/completions"


def build_llama_chat_request(
    payload: Dict[str, Any],
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    include_response_format: bool = DEFAULT_RESPONSE_FORMAT,
) -> Dict[str, Any]:
    request = {
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
    }
    if include_response_format:
        request["response_format"] = {"type": "json_object"}
    return request


def build_openrouter_provider_config(
    provider_order: Sequence[str],
    allow_fallbacks: Optional[bool],
) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    order = [str(provider).strip() for provider in provider_order if str(provider).strip()]
    if order:
        config["order"] = order
    if allow_fallbacks is not None:
        config["allow_fallbacks"] = bool(allow_fallbacks)
    return config


def build_llama_message_content(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    controller_prompt = str(payload.get("controller_prompt", "")).strip() or default_controller_prompt()
    content: list[Dict[str, Any]] = [
        {
            "type": "text",
            "text": controller_prompt + "\n\n" + build_context_text(payload),
        }
    ]
    for frame in latest_frames(payload.get("frames", []), UPSTREAM_IMAGE_COUNT):
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
            "text": (
                "Reply now as normal text. Include at most one {motion: short generic motion prompt} "
                "tag only when the avatar should move."
            ),
        }
    )
    return content


def build_context_text(payload: Dict[str, Any]) -> str:
    context: Dict[str, Any] = {}
    for key in CONTEXT_KEYS:
        value = payload.get(key)
        if key == "world_memory":
            value = compact_world_memory_context(value)
        if has_context_value(value):
            context[key] = value
    return "Decision context JSON:\n" + json.dumps(context, ensure_ascii=True, sort_keys=True)


def compact_world_memory_context(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: Dict[str, Any] = {}
    for key in (
        "current_cell",
        "current_floor_position_m",
        "visited_cell_count",
        "suggested_exploration_goal_location",
        "exploration_hint",
    ):
        if key in value:
            compact[key] = value.get(key)
    compact["nearby_or_recent_cells"] = [
        compact_world_memory_cell(cell)
        for cell in limited_dicts(value.get("nearby_or_recent_cells"), WORLD_MEMORY_CELL_COUNT)
    ]
    compact["candidate_unvisited_goal_locations"] = limited_values(
        value.get("candidate_unvisited_goal_locations"),
        WORLD_MEMORY_FRONTIER_GOAL_COUNT,
    )
    compact["frontier_goal_locations"] = limited_values(
        value.get("frontier_goal_locations"),
        WORLD_MEMORY_FRONTIER_GOAL_COUNT,
    )
    compact["recent_walk_attempts"] = [
        compact_walk_attempt(attempt)
        for attempt in limited_dicts(
            value.get("recent_walk_attempts"),
            WORLD_MEMORY_WALK_ATTEMPT_COUNT,
            from_end=True,
        )
    ]
    return compact


def compact_world_memory_cell(cell: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: cell.get(key)
        for key in ("cell", "center_floor_position_m", "visits", "is_current")
        if key in cell
    }


def compact_walk_attempt(attempt: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: attempt.get(key)
        for key in (
            "prompt",
            "goal_location",
            "latest_distance_to_goal_m",
            "progress_to_goal_m",
            "status",
        )
        if key in attempt
    }


def latest_frames(value: Any, limit: int) -> list[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    frames = [frame for frame in value if isinstance(frame, dict)]
    return frames[-int(limit) :]


def has_context_value(value: Any) -> bool:
    if value is None:
        return False
    if value == "" or value == [] or value == {}:
        return False
    return True


def limited_dicts(value: Any, limit: int, from_end: bool = False) -> list[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    items = list(value)[-limit:] if from_end else list(value)[:limit]
    return [item for item in items if isinstance(item, dict)]


def limited_values(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return list(value)[:limit]


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
    return post_json_with_headers(
        url=url,
        payload=payload,
        timeout_sec=timeout_sec,
        headers={},
        error_context="llama.cpp",
    )


def post_json_with_headers(
    url: str,
    payload: Dict[str, Any],
    timeout_sec: float,
    headers: Dict[str, str],
    error_context: str,
) -> Dict[str, Any]:
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        **headers,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = body.strip() or str(exc)
        raise RuntimeError(
            "{context} request failed with status {status}: {message}".format(
                context=error_context,
                status=exc.code,
                message=message,
            )
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("{context} request failed: {error}".format(context=error_context, error=exc)) from exc
    return decode_json_body(body, error_context)


def get_json(url: str, timeout_sec: float) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError("llama.cpp health request failed: {error}".format(error=exc)) from exc
    return decode_json_body(body, "llama.cpp")


def decode_json_body(body: str, error_context: str = "llama.cpp") -> Dict[str, Any]:
    try:
        decoded = json.loads(body or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("{context} returned invalid JSON: {body}".format(context=error_context, body=body)) from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("{context} returned a non-object JSON response.".format(context=error_context))
    return decoded


def default_controller_prompt() -> str:
    return (
        "You control a VRChat avatar from live screen images. Reply in short natural "
        "speech text. Motion is optional and happens only when you include exactly one "
        "inline tag like {motion: wave hello}. If no movement is useful, omit the tag. "
        "Text outside the tag is speech text for future TTS. DART receives only the "
        "motion tag prompt and cannot see the image, so never mention visible targets "
        "or scene references such as nearest avatar, that person, the object, the "
        "mirror, the door, or towards it in the tag. Use generic motion primitives "
        "such as walk forward, walk backward, turn left, turn right, wave, point "
        "forward, look around, step back, crouch, jump, dance, idle shift, or stand "
        "still. Do not return JSON unless explicitly instructed by the caller."
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


def normalize_model_response(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "")
    tag_payload = extract_motion_tag_payload(text)
    if tag_payload is not None:
        return normalize_action(tag_payload)
    legacy_payload = extract_json_object(text)
    if isinstance(legacy_payload, dict):
        return normalize_action(legacy_payload)
    return {
        "action": "noop",
        "reason": "no_motion_tag",
        "speech_text": strip_motion_tags(text),
    }


def extract_motion_tag_payload(text: str) -> Optional[Dict[str, Any]]:
    matches = list(MOTION_TAG_PATTERN.finditer(str(text)))
    if not matches:
        return None
    speech_text = strip_motion_tags(text)
    for match in matches:
        prompt = str(match.group(1)).strip()
        if prompt:
            return {
                "action": "generate_motion",
                "prompt": prompt,
                "speech_text": speech_text,
                "reason": "motion_tag",
            }
    return {
        "action": "noop",
        "reason": "empty_motion_tag",
        "speech_text": speech_text,
    }


def strip_motion_tags(text: str) -> str:
    stripped = MOTION_TAG_PATTERN.sub(" ", str(text))
    return " ".join(stripped.split()).strip()


def normalize_action(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"action": "noop", "reason": "invalid_json"}
    action = str(payload.get("action", "")).strip()
    action_key = action.lower()
    if action_key == "noop":
        result = {
            "action": "noop",
            "reason": str(payload.get("reason", "")),
        }
        speech_text = str(payload.get("speech_text", ""))
        if speech_text:
            result["speech_text"] = speech_text
        return result
    if action_key == "generate_motion":
        return normalize_generate_motion(payload)
    if is_go_to_action(action):
        prompt = str(payload.get("prompt", "")).strip() or "walk"
        return normalize_generate_motion({**payload, "action": "generate_motion", "prompt": prompt})
    if is_motion_shorthand_action(action):
        prompt = str(payload.get("prompt", "")).strip() or action
        return normalize_generate_motion({**payload, "action": "generate_motion", "prompt": prompt})
    if action_key == "reset_session":
        result = {
            "action": "reset_session",
            "reason": str(payload.get("reason", "")),
        }
        speech_text = str(payload.get("speech_text", ""))
        if speech_text:
            result["speech_text"] = speech_text
        return result
    return {"action": "noop", "reason": "unsupported_action: {action}".format(action=action)}


def normalize_generate_motion(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = normalize_motion_prompt_for_dart(str(payload.get("prompt", "")).strip())
    if not prompt:
        return {"action": "noop", "reason": "missing_prompt"}
    result: Dict[str, Any] = {
        "action": "generate_motion",
        "prompt": prompt,
        "reset_session": bool(payload.get("reset_session", False)),
    }
    speech_text = str(payload.get("speech_text", ""))
    if speech_text:
        result["speech_text"] = speech_text
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


def normalize_motion_prompt_for_dart(prompt: str) -> str:
    normalized = " ".join(str(prompt).strip().lower().replace("_", " ").split())
    normalized = normalized.strip(" .,!?:;\"'")
    if not normalized:
        return ""
    if looks_like_generic_motion_prompt(normalized):
        return normalized
    if re.search(r"\b(turn|rotate|pivot)\s+left\b", normalized):
        return "turn left"
    if re.search(r"\b(turn|rotate|pivot)\s+to\s+the\s+left\b", normalized):
        return "turn left"
    if re.search(r"\b(turn|rotate|pivot)\s+right\b", normalized):
        return "turn right"
    if re.search(r"\b(turn|rotate|pivot)\s+to\s+the\s+right\b", normalized):
        return "turn right"
    if re.search(r"\b(back away|step away|move away|walk away|retreat)\b", normalized):
        return "step back"
    if re.search(r"\b(step|walk|move)\s+back(ward|wards)?\b", normalized):
        return "walk backward"
    if re.search(r"\b(run|jog)\b", normalized):
        return "jog forward"
    if re.search(r"\b(walk|move|go|approach|head|proceed|advance|follow)\b", normalized):
        return "walk forward"
    if re.search(r"\bwave\b", normalized):
        return "wave"
    if re.search(r"\b(point|gesture)\b", normalized):
        return "point forward"
    if re.search(r"\b(look|scan|watch|observe|inspect)\b", normalized):
        return "look around"
    if re.search(r"\b(crouch|duck|squat)\b", normalized):
        return "crouch"
    if re.search(r"\b(jump|hop)\b", normalized):
        return "jump"
    if re.search(r"\b(dance|celebrate)\b", normalized):
        return "dance"
    if re.search(r"\b(idle|shift|fidget)\b", normalized):
        return "idle shift"
    if re.search(r"\b(stand|stop|stay)\b", normalized):
        return "stand still"
    scrubbed = re.sub(
        r"\b(nearest|nearby|closest|that|this|the|a|an)\s+"
        r"(avatar|player|person|character|object|door|doorway|mirror|path|obstacle|wall|chair|thing)\b",
        "",
        normalized,
    )
    scrubbed = re.sub(
        r"\b(towards?|to|at|near|beside|behind|around|through|into|onto|from)\b.*$",
        "",
        scrubbed,
    )
    scrubbed = " ".join(scrubbed.split()).strip(" .,!?:;\"'")
    if scrubbed:
        return " ".join(scrubbed.split()[:6])
    return "look around"


def looks_like_generic_motion_prompt(prompt: str) -> bool:
    if len(prompt.split()) > 8:
        return False
    return bool(GENERIC_MOTION_PATTERN.search(prompt)) and not SCENE_REFERENCE_PATTERN.search(prompt)


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


def parse_provider_order(value: str) -> tuple[str, ...]:
    providers = []
    for part in str(value).replace(";", ",").split(","):
        provider = part.strip()
        if provider:
            providers.append(provider)
    return tuple(providers)


def parse_optional_bool(value: str) -> Optional[bool]:
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("Expected boolean value, got {value!r}.".format(value=value))


def main() -> int:
    args = parse_args()
    if args.backend == "openrouter":
        model_id = args.model_id or os.environ.get("OPENROUTER_MODEL_ID") or DEFAULT_OPENROUTER_MODEL_ID
        provider_order = parse_provider_order(
            args.openrouter_provider
            or os.environ.get("OPENROUTER_PROVIDER", "")
            or os.environ.get("OPENROUTER_PROVIDER_ORDER", "")
        )
        env_allow_fallbacks = parse_optional_bool(os.environ.get("OPENROUTER_ALLOW_FALLBACKS", ""))
        env_response_format = parse_optional_bool(os.environ.get("OPENROUTER_RESPONSE_FORMAT", ""))
        allow_fallbacks = (
            args.openrouter_allow_fallbacks
            if args.openrouter_allow_fallbacks is not None
            else env_allow_fallbacks
            if env_allow_fallbacks is not None
            else False
            if provider_order
            else None
        )
        response_format = (
            args.openrouter_response_format
            if args.openrouter_response_format is not None
            else env_response_format
            if env_response_format is not None
            else DEFAULT_RESPONSE_FORMAT
        )
        backend = OpenRouterVlmBackend(
            model_id=model_id,
            api_key=args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", ""),
            openrouter_url=args.openrouter_url or os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_URL),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            referer=args.openrouter_referer or os.environ.get("OPENROUTER_REFERER", ""),
            title=args.openrouter_title or os.environ.get("OPENROUTER_TITLE", "VRCAI"),
            provider_order=provider_order,
            allow_fallbacks=allow_fallbacks,
            response_format=response_format,
        )
    else:
        model_id = args.model_id or DEFAULT_MODEL_ID
        backend = LlamaCppVlmBackend(
            model_id=model_id,
            llama_url=args.llama_url,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    server = create_http_server(args.host, int(args.port), backend)
    if args.backend == "openrouter":
        print(
            "OpenRouter VLM adapter ready on {host}:{port}; upstream {url}; model {model}".format(
                host=args.host,
                port=args.port,
                url=backend.health()["openrouter_url"],
                model=model_id,
            ),
            flush=True,
        )
    else:
        print(
            "llama.cpp VLM adapter ready on {host}:{port}; upstream {url}; model {model}".format(
                host=args.host,
                port=args.port,
                url=args.llama_url.rstrip("/"),
                model=model_id,
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
