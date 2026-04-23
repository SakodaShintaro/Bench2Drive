# SPDX-License-Identifier: MIT
"""Bench2Drive team-code agent that runs the vla_streaming_rl policy.

The observation/action contract and the route-overlay renderer are reused
verbatim from ``vla_streaming_rl.envs.carla_obs`` so the inputs seen here
match training bit-for-bit.

TEAM_CONFIG is expected to be a directory containing both:
    .hydra/config.yaml   (the Hydra config used for training)
    checkpoint.pt        (trainable state dict produced by scripts/train.py)

Optional env vars:
    VLA_EVAL_VIDEO_PATH   if set, write an mp4 of the composed observation
                          stream (camera + route overlay) to this path.
    VLA_EVAL_VIDEO_FPS    fps for the video (default 20).
"""
import os
from pathlib import Path

import carla
import cv2
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from vla_streaming_rl.agents.streaming import StreamingAgent
from vla_streaming_rl.envs.carla_obs import (
    CARLAObsConfig,
    RouteTracker,
    action_to_vehicle_control,
    camera_sensor_spec,
    compose_obs,
    make_action_space,
    make_obs_space,
)
from vla_streaming_rl.networks.build import build_network


TASK_PROMPT = (
    "Drive a car along a route in CARLA. Follow the planned route, "
    "obey traffic rules, and avoid collisions."
)

# Third-person camera mirrors the training env's spectator cam (carla_leaderboard_env.py)
THIRD_PERSON_SPEC = {
    "type": "sensor.camera.rgb",
    "id": "ThirdPerson",
    "x": -8.0,
    "y": 0.0,
    "z": 5.0,
    "roll": 0.0,
    "pitch": -20.0,
    "yaw": 0.0,
    "width": 800,
    "height": 600,
    "fov": 90,
}


def get_entry_point() -> str:
    return "VLAStreamingAgent"


class VLAStreamingAgent(AutonomousAgent):
    def __init__(self, carla_host, carla_port, debug=False):
        # Bench2Drive calls __init__ -> set_global_plan -> setup (in that order).
        # Anything touched by set_global_plan must be initialised here.
        super().__init__(carla_host, carla_port, debug)
        self.obs_cfg = CARLAObsConfig()
        self.route_tracker: RouteTracker | None = None
        self.prompt = TASK_PROMPT
        self.agent: StreamingAgent | None = None
        self._video_writer = None

    def setup(self, path_to_conf_file: str) -> None:
        self.track = Track.SENSORS

        # leaderboard_evaluator appends "+<save_name>" to the config path
        # (see leaderboard_evaluator.py). Strip the suffix to recover the
        # original TEAM_CONFIG directory.
        base_path = path_to_conf_file.split("+", 1)[0]
        conf_dir = Path(base_path)
        cfg_path = conf_dir / ".hydra" / "config.yaml"
        ckpt_path = conf_dir / "checkpoint.pt"
        if not cfg_path.is_file() or not ckpt_path.is_file():
            raise FileNotFoundError(
                f"expected {cfg_path} and {ckpt_path} under TEAM_CONFIG={conf_dir}"
            )

        cfg = OmegaConf.load(cfg_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_space = make_obs_space(self.obs_cfg)
        act_space = make_action_space()

        network = build_network(
            cfg,
            observation_space_shape=obs_space.shape,
            action_space_shape=act_space.shape,
            parse_action_text=None,
            task_prompt=self.prompt,
            device=device,
            compile=False,
        )
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        missing, unexpected = network.load_state_dict(state, strict=False)
        print(
            f"[vla_streaming_agent] loaded checkpoint: {len(state)} trained params, "
            f"missing={len(missing)}, unexpected={len(unexpected)}",
            flush=True,
        )
        network.eval()

        self.agent = StreamingAgent(
            observation_space=obs_space,
            action_space=act_space,
            network=network,
            normalizing_by_return=bool(cfg.normalizing_by_return),
            max_grad_norm=cfg.max_grad_norm,
            use_done=bool(cfg.use_done),
            accumulation_steps=cfg.accumulation_steps,
            seq_len=cfg.seq_len,
            horizon=cfg.horizon,
            use_eligibility_trace=bool(cfg.use_eligibility_trace),
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            et_lambda=cfg.et_lambda,
            buffer_device=cfg.buffer_device,
            max_new_tokens=cfg.max_new_tokens,
            max_prompt_tokens=cfg.max_prompt_tokens,
            pad_token_id=cfg.pad_token_id,
        )

        video_path = os.environ.get("VLA_EVAL_VIDEO_PATH")
        self._video_writer = None
        if video_path:
            fps = int(os.environ.get("VLA_EVAL_VIDEO_FPS", "20"))
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            self._video_writer = imageio.get_writer(
                video_path, fps=fps, macro_block_size=1
            )
            print(f"[vla_streaming_agent] writing video to {video_path} @ {fps}fps",
                  flush=True)

    def sensors(self) -> list[dict]:
        specs = [camera_sensor_spec(self.obs_cfg, sensor_id="Center")]
        if self._video_writer is not None:
            specs.append(THIRD_PERSON_SPEC)
        return specs

    def set_global_plan(self, global_plan_gps, global_plan_world_coord) -> None:
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        # Use the *full*, non-downsampled route (parent downsamples to ~every
        # 50m); interpolation to a fixed density happens inside RouteTracker.
        raw = [
            carla.Location(t.location.x, t.location.y, t.location.z)
            for t, _ in global_plan_world_coord
        ]
        self.route_tracker = RouteTracker.from_raw_waypoints(raw, self.obs_cfg)

    def run_step(self, input_data: dict, timestamp: float) -> carla.VehicleControl:
        # Camera: BGRA -> HWC uint8 RGB (mirrors training env's _process_image)
        bgra = input_data["Center"][1]
        rgb_hwc = bgra[:, :, [2, 1, 0]]

        assert self.route_tracker is not None, "set_global_plan must run before run_step"
        vehicle_loc = self.hero_actor.get_location()
        self.route_tracker.update(vehicle_loc)
        yaw_rad = np.radians(self.hero_actor.get_transform().rotation.yaw)
        overlay = self.route_tracker.render_overlay(vehicle_loc, yaw_rad)

        obs = compose_obs(rgb_hwc, overlay, self.obs_cfg)  # (3, H, W) float32 [0,1]

        action, _ = self.agent.select_action(
            global_step=0,
            obs=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            task_prompt=self.prompt,
        )

        if self._video_writer is not None:
            self._video_writer.append_data(
                self._build_video_frame(obs, input_data.get("ThirdPerson"))
            )

        steer, throttle, brake = action_to_vehicle_control(action)
        return carla.VehicleControl(
            steer=steer,
            throttle=throttle,
            brake=brake,
            hand_brake=False,
            manual_gear_shift=False,
        )

    def _build_video_frame(
        self, obs_chw_float: np.ndarray, third_person_entry: tuple | None
    ) -> np.ndarray:
        """Return an HWC uint8 RGB frame: ``[third_person | observation]``.

        The observation (256x256) is upscaled with nearest-neighbor to match
        the third-person camera height (600), keeping pixels crisp.
        """
        obs_rgb = (obs_chw_float.transpose(1, 2, 0) * 255).astype(np.uint8)
        if third_person_entry is None:
            return obs_rgb

        tp_bgra = third_person_entry[1]
        tp_rgb = tp_bgra[:, :, [2, 1, 0]]
        tp_h = tp_rgb.shape[0]
        obs_up = cv2.resize(obs_rgb, (tp_h, tp_h), interpolation=cv2.INTER_NEAREST)
        return np.concatenate([tp_rgb, obs_up], axis=1)

    def destroy(self) -> None:
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        if hasattr(self, "agent"):
            del self.agent
        torch.cuda.empty_cache()
