import argparse
import json
import math
import os
import pickle
import random
import socketserver
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vrcai.dart_conversion import convert_dart_world_joints_to_raw_positions, write_mdm_style_results


SERVICE_NAME = "dart_session_service_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent DART prompt generation service.")
    parser.add_argument("--host", default="127.0.0.1", help="Service bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Service TCP port.")
    parser.add_argument("--dart-dir", default=str(ROOT / "dart"), help="Path to the DART checkout.")
    parser.add_argument("--denoiser-checkpoint", required=True, help="Path to the DART denoiser checkpoint.")
    parser.add_argument("--dataset", default="babel", help="DART rollout dataset name.")
    parser.add_argument("--device", default="cuda", help="Torch device to use.")
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size. Session mode currently supports 1.")
    parser.add_argument("--respacing", default="", help="Optional diffusion respacing value.")
    parser.add_argument("--use-predicted-joints", type=int, default=1, help="Use predicted joints as in run_demo.")
    parser.add_argument("--zero-noise", type=int, default=0, help="Use zero init noise for sampling.")
    parser.add_argument("--show-viewer", action="store_true", help="Visualize the latest generated segment on the generator machine.")
    parser.add_argument("--max-frame-count", type=int, default=196, help="Maximum allowed frame count.")
    return parser.parse_args()


@dataclass
class ServiceState:
    generator: "DartSessionGenerator"
    lock: threading.Lock


class ViewerController:
    def __init__(self, python_executable: str, dart_dir: Path) -> None:
        self._python_executable = python_executable
        self._dart_dir = dart_dir
        self._process: Optional[subprocess.Popen[bytes]] = None

    def show(self, seq_path: Path) -> None:
        self.close()
        command = [
            self._python_executable,
            "-m",
            "visualize.vis_seq",
            "--add_floor",
            "1",
            "--translate_body",
            "1",
            "--seq_path",
            str(seq_path),
        ]
        self._process = subprocess.Popen(
            command,
            cwd=str(self._dart_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        self._process = None


class DartSessionGenerator:
    def __init__(self, args: argparse.Namespace) -> None:
        if int(args.batch_size) != 1:
            raise ValueError("DART session mode currently supports --batch-size 1.")

        self.args = args
        self.dart_dir = Path(args.dart_dir).expanduser().resolve()
        if str(self.dart_dir) not in sys.path:
            sys.path.insert(0, str(self.dart_dir))

        os.chdir(self.dart_dir)

        from data_loaders.humanml.data.dataset import SinglePrimitiveDataset
        from mld.rollout_mld import load_mld
        from mld.train_mld import create_gaussian_diffusion
        from utils.misc_util import encode_text
        from utils.smpl_utils import PrimitiveUtility

        self._encode_text = encode_text
        self._SinglePrimitiveDataset = SinglePrimitiveDataset
        self._PrimitiveUtility = PrimitiveUtility

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.set_default_dtype(torch.float32)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.args.device = str(self.device)

        self.denoiser_args, self.denoiser_model, self.vae_args, self.vae_model = load_mld(
            args.denoiser_checkpoint,
            self.device,
        )
        diffusion_args = self.denoiser_args.diffusion_args
        diffusion_args.respacing = args.respacing
        self.diffusion = create_gaussian_diffusion(diffusion_args)
        self.sample_fn = self.diffusion.p_sample_loop if args.respacing == "" else self.diffusion.ddim_sample_loop

        self.dataset = self._SinglePrimitiveDataset(
            cfg_path=self.vae_args.data_args.cfg_path,
            dataset_path=self.vae_args.data_args.data_dir,
            body_type=self.vae_args.data_args.body_type,
            sequence_path="./data/stand.pkl" if args.dataset == "babel" else "./data/stand_20fps.pkl",
            batch_size=args.batch_size,
            device=self.device,
            enforce_gender="male",
            enforce_zero_beta=1,
        )
        self.primitive_utility = self._PrimitiveUtility(
            device=self.device,
            dtype=torch.float32,
            body_type=self.vae_args.data_args.body_type,
        )
        self.future_length = int(self.dataset.future_length)
        self.history_length = int(self.dataset.history_length)
        self.batch_size = int(args.batch_size)
        self.viewer = ViewerController(sys.executable, self.dart_dir) if args.show_viewer else None

        batch = self.dataset.get_batch(batch_size=self.batch_size)
        seed_batch = batch[0]
        input_motions = seed_batch["motion_tensor_normalized"].to(self.device)
        self.gender = seed_batch["gender"][0]
        primitive_length = self.history_length + self.future_length
        self.betas = seed_batch["betas"][:, :primitive_length, :].to(self.device)
        self.pelvis_delta = self.primitive_utility.calc_calibrate_offset(
            {
                "betas": self.betas[:, 0, :],
                "gender": self.gender,
            }
        )
        self.initial_history_motion = input_motions.squeeze(2).permute(0, 2, 1)[:, : self.history_length, :].clone()
        self.initial_transf_rotmat = torch.eye(3, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(
            self.batch_size, 1, 1
        )
        self.initial_transf_transl = torch.zeros(
            self.batch_size, 1, 3, device=self.device, dtype=torch.float32
        )
        self.reset()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()

    def reset(self) -> None:
        self.history_motion = self.initial_history_motion.clone()
        self.transf_rotmat = self.initial_transf_rotmat.clone()
        self.transf_transl = self.initial_transf_transl.clone()

    def generate_motion(
        self,
        prompt: str,
        frame_count: int,
        guidance_param: float,
        output_dir: Path,
        seed: Optional[int] = None,
        reset_session: bool = False,
    ) -> Dict[str, Any]:
        if reset_session:
            self.reset()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            torch.manual_seed(seed)

        text_embedding = self._encode_text(
            self.dataset.clip_model,
            [prompt],
            force_empty_zero=True,
        ).to(dtype=torch.float32, device=self.device)

        remaining = int(frame_count)
        world_joint_segments = []
        viewer_sequences: Optional[dict[str, list[torch.Tensor]]] = {
            "texts": [prompt],
            "transl": [],
            "global_orient": [],
            "body_pose": [],
            "joints": [],
        }

        while remaining > 0:
            guidance = torch.ones(
                self.batch_size,
                *self.denoiser_args.model_args.noise_shape,
                device=self.device,
            ) * float(guidance_param)
            y = {
                "text_embedding": text_embedding.expand(self.batch_size, -1),
                "history_motion_normalized": self.history_motion,
                "scale": guidance,
            }

            x_start_pred = self.sample_fn(
                self.denoiser_model,
                (self.batch_size, *self.denoiser_args.model_args.noise_shape),
                clip_denoised=False,
                model_kwargs={"y": y},
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=torch.zeros_like(guidance) if int(self.args.zero_noise) else None,
                const_noise=False,
            )
            latent_pred = x_start_pred.permute(1, 0, 2)
            future_motion_pred = self.vae_model.decode(
                latent_pred,
                self.history_motion,
                nfuture=self.future_length,
                scale_latent=self.denoiser_args.rescale_latent,
            )
            future_frames = self.dataset.denormalize(future_motion_pred)

            accepted = min(remaining, self.future_length)
            accepted_future_frames = future_frames[:, :accepted, :]
            world_feature_dict = self._future_frames_to_world_feature_dict(accepted_future_frames)
            world_joints = world_feature_dict["joints"].reshape(self.batch_size, accepted, 22, 3)
            world_joint_segments.append(world_joints[0].detach().cpu().numpy())

            viewer_primitive = self.primitive_utility.feature_dict_to_smpl_dict(world_feature_dict)
            viewer_sequences["transl"].append(viewer_primitive["transl"][0].detach().cpu())
            viewer_sequences["global_orient"].append(viewer_primitive["global_orient"][0].detach().cpu())
            viewer_sequences["body_pose"].append(viewer_primitive["body_pose"][0].detach().cpu())
            viewer_sequences["joints"].append(world_joints[0].detach().cpu())

            self._advance_session_state(accepted_future_frames)
            remaining -= accepted

        world_joints = np.concatenate(world_joint_segments, axis=0)
        raw_positions = convert_dart_world_joints_to_raw_positions(world_joints)
        write_mdm_style_results(output_dir, raw_positions, prompt=prompt, generator=SERVICE_NAME)

        viewer_sample_path: Optional[Path] = None
        if self.viewer is not None and viewer_sequences is not None:
            viewer_sample_path = self._write_viewer_sample(output_dir, viewer_sequences)
            self.viewer.show(viewer_sample_path)

        return {
            "frame_count": int(raw_positions.shape[0]),
            "joint_count": int(raw_positions.shape[1]),
            "viewer_sample": str(viewer_sample_path) if viewer_sample_path is not None else "",
        }

    def _future_frames_to_world_feature_dict(self, future_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        accepted = int(future_frames.shape[1])
        future_feature_dict = self.primitive_utility.tensor_to_dict(future_frames)
        future_feature_dict.update(
            {
                "transf_rotmat": self.transf_rotmat,
                "transf_transl": self.transf_transl,
                "gender": self.gender,
                "betas": self.betas[:, :accepted, :],
                "pelvis_delta": self.pelvis_delta,
            }
        )
        return self.primitive_utility.transform_feature_to_world(future_feature_dict)

    def _advance_session_state(self, accepted_future_frames: torch.Tensor) -> None:
        history_frames = self.dataset.denormalize(self.history_motion)
        all_frames = torch.cat([history_frames, accepted_future_frames], dim=1)
        new_history_frames = all_frames[:, -self.history_length :, :]
        history_feature_dict = self.primitive_utility.tensor_to_dict(new_history_frames)
        history_feature_dict.update(
            {
                "transf_rotmat": self.transf_rotmat,
                "transf_transl": self.transf_transl,
                "gender": self.gender,
                "betas": self.betas[:, : self.history_length, :],
                "pelvis_delta": self.pelvis_delta,
            }
        )
        canonicalized_history, blended_feature_dict = self.primitive_utility.get_blended_feature(
            history_feature_dict,
            use_predicted_joints=bool(self.args.use_predicted_joints),
        )
        self.transf_rotmat = canonicalized_history["transf_rotmat"]
        self.transf_transl = canonicalized_history["transf_transl"]
        self.history_motion = self.dataset.normalize(self.primitive_utility.dict_to_tensor(blended_feature_dict))

    def _write_viewer_sample(self, output_dir: Path, viewer_sequences: dict[str, list[torch.Tensor]]) -> Path:
        seq_path = Path(output_dir) / "sample_0.pkl"
        sequence = {
            "texts": viewer_sequences["texts"],
            "gender": self.gender,
            "betas": self.betas[0, 0, :10].detach().cpu(),
            "transl": torch.cat(viewer_sequences["transl"], dim=0),
            "global_orient": torch.cat(viewer_sequences["global_orient"], dim=0),
            "body_pose": torch.cat(viewer_sequences["body_pose"], dim=0),
            "joints": torch.cat(viewer_sequences["joints"], dim=0),
            "history_length": self.history_length,
            "future_length": self.future_length,
        }
        with seq_path.open("wb") as handle:
            pickle.dump(sequence, handle)
        return seq_path


class PromptRequestHandler(socketserver.StreamRequestHandler):
    state: ServiceState
    max_frame_count: int

    def handle(self) -> None:
        line = self.rfile.readline()
        if not line:
            return
        try:
            request = json.loads(line.decode("utf-8"))
            response = self._process_request(request)
        except Exception as exc:  # noqa: BLE001
            response = {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        self.wfile.write((json.dumps(response, ensure_ascii=True) + "\n").encode("utf-8"))

    def _process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        task = str(request.get("task", "generate_motion"))
        if task != "generate_motion":
            return {"ok": False, "error": f"Unsupported task '{task}'."}

        prompt = str(request.get("prompt", "")).strip()
        if not prompt:
            return {"ok": False, "error": "Request field 'prompt' is required."}

        frame_count = int(request.get("frame_count", 0))
        if frame_count <= 0:
            return {"ok": False, "error": "Request field 'frame_count' must be > 0."}
        if frame_count > self.max_frame_count:
            return {
                "ok": False,
                "error": f"frame_count {frame_count} exceeds max_frame_count {self.max_frame_count}.",
            }

        output_dir = str(request.get("output_dir", "")).strip()
        if not output_dir:
            return {"ok": False, "error": "Request field 'output_dir' is required."}

        guidance_param = float(request.get("guidance_param", 5.0))
        seed_value = request.get("seed")
        seed = int(seed_value) if seed_value is not None else None
        reset_session = bool(request.get("reset_session", False))

        with self.state.lock:
            result = self.state.generator.generate_motion(
                prompt=prompt,
                frame_count=frame_count,
                guidance_param=guidance_param,
                output_dir=Path(output_dir),
                seed=seed,
                reset_session=reset_session,
            )

        return {
            "ok": True,
            "task": task,
            "output_dir": output_dir,
            "frame_count": result["frame_count"],
            "joint_count": result["joint_count"],
            "seed": seed,
            "guidance_param": guidance_param,
            "reset_session": reset_session,
            "service": SERVICE_NAME,
            "viewer_sample": result["viewer_sample"],
        }


def _build_handler(state: ServiceState, max_frame_count: int):
    class _Handler(PromptRequestHandler):
        pass

    _Handler.state = state
    _Handler.max_frame_count = max_frame_count
    return _Handler


def main() -> int:
    args = parse_args()
    generator = DartSessionGenerator(args)
    state = ServiceState(generator=generator, lock=threading.Lock())
    handler_class = _build_handler(state=state, max_frame_count=int(args.max_frame_count))
    try:
        with socketserver.ThreadingTCPServer((args.host, int(args.port)), handler_class) as server:
            server.daemon_threads = True
            print(
                "DART session service ready on {host}:{port}".format(
                    host=args.host,
                    port=args.port,
                ),
                flush=True,
            )
            server.serve_forever()
    finally:
        generator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
