from __future__ import annotations

import argparse
from base64 import b64decode
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from typing import Any, Dict, Optional


SERVICE_NAME = "gemma_vlm_service_v1"
DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8777
DEFAULT_MAX_NEW_TOKENS = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Gemma VLM decision service.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Service bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Service TCP port.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


@dataclass
class GemmaVlmBackend:
    model_id: str = DEFAULT_MODEL_ID
    device_map: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS

    def __post_init__(self) -> None:
        self._processor = None
        self._model = None
        self._torch = None

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._load()
        images = self._decode_images(payload)
        messages = [
            {
                "role": "user",
                "content": self._build_content(payload=payload, images=images),
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if hasattr(self._model, "device"):
            inputs = inputs.to(self._model.device)
        output = self._model.generate(
            **inputs,
            max_new_tokens=int(self.max_new_tokens),
        )
        prompt_token_count = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        generated_tokens = output[0][prompt_token_count:] if prompt_token_count else output[0]
        raw_text = self._processor.decode(generated_tokens, skip_special_tokens=True).strip()
        if not raw_text:
            raw_text = self._processor.decode(output[0], skip_special_tokens=True).strip()
        return {
            "ok": True,
            "service": SERVICE_NAME,
            "model_id": self.model_id,
            "raw_text": raw_text,
            "action": normalize_action(extract_json_object(raw_text)),
        }

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Gemma VLM service requires torch and transformers. "
                "Install the VLM extras in the remote environment."
            ) from exc

        dtype = None
        if self.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.dtype == "float16":
            dtype = torch.float16
        elif self.dtype == "float32":
            dtype = torch.float32

        kwargs: Dict[str, Any] = {"device_map": self.device_map}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_id, padding_side="left")
        self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)

    def _decode_images(self, payload: Dict[str, Any]) -> list[Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Gemma VLM service requires Pillow for image decoding.") from exc

        images = []
        for frame in payload.get("frames", []):
            if not isinstance(frame, dict):
                continue
            encoded = str(frame.get("image_base64", "")).strip()
            if not encoded:
                data_url = str(frame.get("data_url", ""))
                if "," in data_url:
                    encoded = data_url.split(",", 1)[1]
            if not encoded:
                continue
            image_bytes = b64decode(encoded)
            images.append(Image.open(BytesIO(image_bytes)).convert("RGB"))
        return images

    def _build_content(self, payload: Dict[str, Any], images: list[Any]) -> list[Dict[str, Any]]:
        controller_prompt = str(payload.get("controller_prompt", "")).strip() or default_controller_prompt()
        recent_actions = payload.get("recent_actions", [])
        content: list[Dict[str, Any]] = [
            {
                "type": "text",
                "text": controller_prompt
                + "\nRecent actions JSON:\n"
                + json.dumps(recent_actions, ensure_ascii=True),
            }
        ]
        for image in images:
            content.append({"type": "image", "image": image})
        content.append(
            {
                "type": "text",
                "text": "Choose the next action from the newest frame context. Return JSON only.",
            }
        )
        return content


class GemmaVlmRequestHandler(BaseHTTPRequestHandler):
    backend: Any = None

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(404, {"ok": False, "error": "not_found"})
            return
        self._write_json(200, {"ok": True, "service": SERVICE_NAME})

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
        "Make motion prompts concrete and physical, such as walk forward, turn, "
        "wave, look around, step back, point, crouch, or idle shift. Do not choose "
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
    if action == "noop":
        return {
            "action": "noop",
            "reason": str(payload.get("reason", "")),
        }
    if action == "generate_motion":
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
        return result
    if action == "reset_session":
        return {
            "action": "reset_session",
            "reason": str(payload.get("reason", "")),
        }
    return {"action": "noop", "reason": "unsupported_action: {action}".format(action=action)}


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
    backend = GemmaVlmBackend(
        model_id=args.model_id,
        device_map=args.device_map,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    server = create_http_server(args.host, int(args.port), backend)
    print(
        "Gemma VLM service ready on {host}:{port} using {model}".format(
            host=args.host,
            port=args.port,
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
