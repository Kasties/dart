from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Dict, Iterator, Optional

import numpy as np


SERVICE_NAME = "depth_anything_service_v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8779
DEFAULT_BACKEND = "hf"
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_DA3_MODEL_ID = "depth-anything/DA3-SMALL"
DEFAULT_DEVICE = "cuda"
DEFAULT_INPUT_SIZE = 518


@dataclass(frozen=True)
class ImageInput:
    path: Optional[Path]
    image_bytes: Optional[bytes]
    mime_type: str = "image/png"


@dataclass(frozen=True)
class DepthPrediction:
    depth: np.ndarray
    confidence: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Depth Anything HTTP adapter.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Adapter bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Adapter TCP port.")
    parser.add_argument(
        "--backend",
        choices=("hf", "da2", "da3", "custom"),
        default=DEFAULT_BACKEND,
        help="Depth backend to load. hf uses Transformers; da2/da3 use official repositories.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or DA3 pretrained model id.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device, for example cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--repo-dir",
        default="",
        help="Optional local Depth Anything repository path to prepend to sys.path.",
    )
    parser.add_argument("--checkpoint", default="", help="Depth Anything V2 checkpoint path for --backend da2.")
    parser.add_argument(
        "--encoder",
        choices=("vits", "vitb", "vitl", "vitg"),
        default="vits",
        help="Depth Anything V2 encoder for --backend da2.",
    )
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE, help="Depth Anything V2 input size.")
    parser.add_argument(
        "--custom-factory",
        default="",
        help=(
            "Optional module:function for --backend custom. The function receives argparse.Namespace "
            "and must return an object with estimate(image_input, request_payload)."
        ),
    )
    return parser.parse_args()


class TransformersDepthBackend:
    def __init__(self, model_id: str, device: str) -> None:
        self.model_id = model_id
        self.device = device
        self._pipeline: Optional[Any] = None

    def health(self) -> Dict[str, Any]:
        return {"backend": "hf", "model_id": self.model_id, "device": self.device}

    def estimate(self, image_input: ImageInput, request_payload: Dict[str, Any]) -> DepthPrediction:
        pipe = self._load()
        if image_input.path is not None:
            model_input: Any = str(image_input.path)
        else:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("Pillow is required to decode base64 images for the hf backend.") from exc
            if image_input.image_bytes is None:
                raise ValueError("No image bytes were supplied.")
            model_input = Image.open(io.BytesIO(image_input.image_bytes)).convert("RGB")

        result = pipe(model_input)
        depth = _depth_array_from_transformers_result(result)
        return DepthPrediction(
            depth=depth,
            metadata={"backend": "hf", "model_id": self.model_id, "unit": "relative"},
        )

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Transformers is required for --backend hf.") from exc
        self._pipeline = pipeline(
            task="depth-estimation",
            model=self.model_id,
            device=_transformers_device_arg(self.device),
        )
        return self._pipeline


class DepthAnythingV2Backend:
    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(
        self,
        encoder: str,
        checkpoint: str,
        device: str,
        input_size: int,
        repo_dir: str = "",
    ) -> None:
        self.encoder = encoder
        self.checkpoint = checkpoint
        self.device = device
        self.input_size = int(input_size)
        self.repo_dir = repo_dir
        self._model: Optional[Any] = None

    def health(self) -> Dict[str, Any]:
        return {
            "backend": "da2",
            "encoder": self.encoder,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "input_size": self.input_size,
        }

    def estimate(self, image_input: ImageInput, request_payload: Dict[str, Any]) -> DepthPrediction:
        model = self._load()
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for --backend da2.") from exc
        with materialized_image_path(image_input) as image_path:
            raw_img = cv2.imread(str(image_path))
            if raw_img is None:
                raise ValueError("OpenCV could not decode the input image.")
            depth = model.infer_image(raw_img, int(request_payload.get("input_size", self.input_size)))
        return DepthPrediction(
            depth=np.asarray(depth, dtype=np.float32),
            metadata={"backend": "da2", "encoder": self.encoder, "unit": "relative"},
        )

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.checkpoint:
            raise RuntimeError("--backend da2 requires --checkpoint pointing to depth_anything_v2_<encoder>.pth.")
        _prepend_repo_dir(self.repo_dir)
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as exc:
            raise RuntimeError("Depth-Anything-V2 and torch are required for --backend da2.") from exc
        model = DepthAnythingV2(**self.MODEL_CONFIGS[self.encoder])
        model.load_state_dict(torch.load(self.checkpoint, map_location="cpu"))
        self._model = model.to(self.device).eval()
        return self._model


class DepthAnything3Backend:
    def __init__(self, model_id: str, device: str, repo_dir: str = "") -> None:
        self.model_id = model_id or DEFAULT_DA3_MODEL_ID
        self.device = device
        self.repo_dir = repo_dir
        self._model: Optional[Any] = None

    def health(self) -> Dict[str, Any]:
        return {"backend": "da3", "model_id": self.model_id, "device": self.device}

    def estimate(self, image_input: ImageInput, request_payload: Dict[str, Any]) -> DepthPrediction:
        model = self._load()
        with materialized_image_path(image_input) as image_path:
            prediction = model.inference([str(image_path)])
        depth = np.asarray(prediction.depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[0]
        confidence = getattr(prediction, "conf", None)
        if confidence is not None:
            confidence = np.asarray(confidence, dtype=np.float32)
            if confidence.ndim == 3:
                confidence = confidence[0]
        return DepthPrediction(
            depth=depth,
            confidence=confidence,
            metadata={
                "backend": "da3",
                "model_id": self.model_id,
                "unit": "meters" if "metric" in self.model_id.lower() else "relative",
            },
        )

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        _prepend_repo_dir(self.repo_dir)
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise RuntimeError("Depth-Anything-3 is required for --backend da3.") from exc
        model = DepthAnything3.from_pretrained(self.model_id)
        self._model = model.to(device=self.device)
        return self._model


def create_backend(args: argparse.Namespace) -> Any:
    if args.backend == "hf":
        return TransformersDepthBackend(model_id=args.model_id, device=args.device)
    if args.backend == "da2":
        return DepthAnythingV2Backend(
            encoder=args.encoder,
            checkpoint=args.checkpoint,
            device=args.device,
            input_size=args.input_size,
            repo_dir=args.repo_dir,
        )
    if args.backend == "da3":
        model_id = args.model_id
        if model_id == DEFAULT_MODEL_ID:
            model_id = DEFAULT_DA3_MODEL_ID
        return DepthAnything3Backend(model_id=model_id, device=args.device, repo_dir=args.repo_dir)
    if args.backend == "custom":
        if not args.custom_factory:
            raise ValueError("--backend custom requires --custom-factory module:function.")
        return load_custom_backend(args.custom_factory, args)
    raise ValueError("Unsupported backend: {backend}".format(backend=args.backend))


class DepthAnythingRequestHandler(BaseHTTPRequestHandler):
    backend: Any = None

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(404, {"ok": False, "error": "not_found"})
            return
        health = {}
        if hasattr(self.backend, "health"):
            health = self.backend.health()
        self._write_json(200, {"ok": True, "service": SERVICE_NAME, **health})

    def do_POST(self) -> None:
        if self.path != "/estimate":
            self._write_json(404, {"ok": False, "error": "not_found"})
            return
        try:
            payload = self._read_json_body()
            image_input = image_input_from_payload(payload)
            prediction = self.backend.estimate(image_input, payload)
            self._write_json(200, build_estimate_response(prediction, payload))
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
    class _Handler(DepthAnythingRequestHandler):
        pass

    _Handler.backend = backend
    return ThreadingHTTPServer((host, int(port)), _Handler)


def build_estimate_response(prediction: DepthPrediction, payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(prediction.metadata or {})
    response: Dict[str, Any] = {
        "ok": True,
        "service": SERVICE_NAME,
        "depth": depth_stats(prediction.depth, unit=str(metadata.get("unit", "relative"))),
        "metadata": metadata,
    }
    if prediction.confidence is not None:
        response["confidence"] = depth_stats(prediction.confidence, unit="confidence")
    if bool(payload.get("return_depth_png", True)):
        response["depth_png_base64"] = base64.b64encode(encode_depth_png(prediction.depth)).decode("ascii")
    if bool(payload.get("return_depth_npy", False)):
        response["depth_npy_base64"] = base64.b64encode(encode_depth_npy(prediction.depth)).decode("ascii")
    return response


def image_input_from_payload(payload: Dict[str, Any]) -> ImageInput:
    image_path = str(payload.get("image_path", "")).strip()
    if image_path:
        return ImageInput(path=Path(image_path).expanduser(), image_bytes=None)

    data_url = str(payload.get("data_url", "")).strip()
    if data_url:
        mime_type, encoded = split_data_url(data_url)
        return ImageInput(path=None, image_bytes=base64.b64decode(encoded), mime_type=mime_type)

    encoded = str(payload.get("image_base64", "")).strip()
    if encoded:
        mime_type = str(payload.get("mime_type", "image/png")).strip() or "image/png"
        return ImageInput(path=None, image_bytes=base64.b64decode(encoded), mime_type=mime_type)

    raise ValueError("Request must include image_path, data_url, or image_base64.")


def split_data_url(data_url: str) -> tuple[str, str]:
    if not data_url.startswith("data:") or "," not in data_url:
        raise ValueError("Invalid data_url image payload.")
    header, encoded = data_url.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] or "image/png"
    return mime_type, encoded


@contextmanager
def materialized_image_path(image_input: ImageInput) -> Iterator[Path]:
    if image_input.path is not None:
        yield image_input.path
        return
    suffix = ".jpg" if image_input.mime_type.lower() in {"image/jpeg", "image/jpg"} else ".png"
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        if image_input.image_bytes is None:
            raise ValueError("No image bytes were supplied.")
        handle.write(image_input.image_bytes)
        handle.close()
        yield Path(handle.name)
    finally:
        try:
            Path(handle.name).unlink()
        except OSError:
            pass


def depth_stats(depth: np.ndarray, unit: str) -> Dict[str, Any]:
    array = np.asarray(depth, dtype=np.float32)
    finite = array[np.isfinite(array)]
    stats: Dict[str, Any] = {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "unit": unit,
        "finite_count": int(finite.size),
    }
    if finite.size:
        stats.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
            }
        )
    else:
        stats.update({"min": None, "max": None, "mean": None})
    return stats


def encode_depth_png(depth: np.ndarray) -> bytes:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to return depth_png_base64.") from exc
    image = Image.fromarray(normalize_depth_to_uint16(depth), mode="I;16")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def encode_depth_npy(depth: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(depth, dtype=np.float32))
    return buffer.getvalue()


def normalize_depth_to_uint16(depth: np.ndarray) -> np.ndarray:
    array = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros(array.shape, dtype=np.uint16)
    finite_values = array[finite]
    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint16)
    normalized = np.zeros(array.shape, dtype=np.float32)
    normalized[finite] = (array[finite] - min_value) / (max_value - min_value)
    return np.clip(normalized * 65535.0, 0.0, 65535.0).astype(np.uint16)


def _depth_array_from_transformers_result(result: Any) -> np.ndarray:
    if not isinstance(result, dict):
        raise RuntimeError("Transformers depth pipeline returned a non-object result.")
    predicted = result.get("predicted_depth")
    if predicted is not None:
        if hasattr(predicted, "detach"):
            predicted = predicted.detach().cpu().numpy()
        array = np.asarray(predicted, dtype=np.float32)
        return np.squeeze(array)
    depth_image = result.get("depth")
    if depth_image is None:
        raise RuntimeError("Transformers depth pipeline returned no depth output.")
    return np.asarray(depth_image, dtype=np.float32)


def _transformers_device_arg(device: str) -> int:
    normalized = str(device).lower().strip()
    if normalized in {"", "cpu"}:
        return -1
    if normalized.startswith("cuda:"):
        try:
            return int(normalized.split(":", 1)[1])
        except ValueError:
            return 0
    if normalized == "cuda":
        return 0
    return -1


def _prepend_repo_dir(repo_dir: str) -> None:
    if not repo_dir:
        return
    path = str(Path(repo_dir).expanduser().resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def load_custom_backend(factory_ref: str, args: argparse.Namespace) -> Any:
    if ":" not in factory_ref:
        raise ValueError("--custom-factory must be formatted as module:function.")
    module_name, function_name = factory_ref.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    backend = factory(args)
    if not hasattr(backend, "estimate"):
        raise TypeError("Custom depth backend must provide estimate(image_input, request_payload).")
    return backend


def main() -> int:
    args = parse_args()
    backend = create_backend(args)
    server = create_http_server(args.host, args.port, backend)
    print(
        "Depth Anything adapter ready on {host}:{port}; backend {backend}; model {model}".format(
            host=args.host,
            port=args.port,
            backend=args.backend,
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
