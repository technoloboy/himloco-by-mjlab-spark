"""Interactive play script with gait stability monitoring.

Extends play_keyboard.py by recording per-step data and producing
a detailed gait analysis report on exit.

Controls (same as play_keyboard.py):
  w / s  — increase / decrease forward velocity (vx)
  a / d  — strafe left / right (vy)
  q / e  — turn left / right (wz)
  x      — zero all velocity commands
  9      — reset environment + zero commands
  m      — print live gait stats snapshot
  ESC / Ctrl-C — quit + save analysis

Output files (written to logs/gait_monitor/):
  gait_data_<timestamp>.npz  — raw recorded data
  gait_report_<timestamp>.png — analysis figures
"""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro


import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.terrains import (
    BoxPyramidStairsTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    TerrainGeneratorCfg,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

# ── Terrain (same as play_keyboard.py) ───────────────────────────────────────
_KEYBOARD_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=5,
    num_cols=10,
    curriculum=False,
    sub_terrains={
        "pyramid_stairs": BoxPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
        ),
        "hf_discrete_obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.5,
            obstacle_height_range=(0.05, 0.12),
            obstacle_width_range=(0.4, 0.8),
            num_obstacles=20,
            platform_width=1.0,
        ),
    },
    add_lights=True,
)

_VX_MAX = 2.0
_VY_MAX = 1.0
_WZ_MAX = 1.0
_FREEZE  = 1e6

# ── Joint / actuator metadata ─────────────────────────────────────────────────
# Order matches MJCF actuator definition: FR, FL, RR, RL × (hip, thigh, calf)
JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]
# Feet contact sensor order: FR, FL, RR, RL  (matches sensor config)
FOOT_NAMES = ["FR", "FL", "RR", "RL"]
NUM_JOINTS = len(JOINT_NAMES)
NUM_FEET   = len(FOOT_NAMES)


# ── Gait data recorder ────────────────────────────────────────────────────────

class GaitRecorder:
    """Accumulates per-step state tensors for offline gait analysis."""

    def __init__(self, max_steps: int = 100_000) -> None:
        self._max = max_steps
        # Each list entry is a 1-D numpy array captured at one env step.
        self.actions:      list[np.ndarray] = []   # (12,)  policy output
        self.joint_pos:    list[np.ndarray] = []   # (12,)  qpos joint part
        self.joint_vel:    list[np.ndarray] = []   # (12,)  qvel joint part
        self.body_pos:     list[np.ndarray] = []   # (3,)   x,y,z
        self.body_quat:    list[np.ndarray] = []   # (4,)   w,x,y,z (MuJoCo)
        self.body_lin_vel: list[np.ndarray] = []   # (3,)   world frame
        self.body_ang_vel: list[np.ndarray] = []   # (3,)   world frame
        self.foot_contact: list[np.ndarray] = []   # (4,)   binary
        self.command:      list[np.ndarray] = []   # (3,)   vx,vy,wz
        self.timestamps:   list[float]      = []   # wall-clock seconds

    def record(
        self,
        actions:      np.ndarray,
        joint_pos:    np.ndarray,
        joint_vel:    np.ndarray,
        body_pos:     np.ndarray,
        body_quat:    np.ndarray,
        body_lin_vel: np.ndarray,
        body_ang_vel: np.ndarray,
        foot_contact: np.ndarray,
        command:      np.ndarray,
    ) -> None:
        if len(self.timestamps) >= self._max:
            return
        self.actions.append(actions.copy())
        self.joint_pos.append(joint_pos.copy())
        self.joint_vel.append(joint_vel.copy())
        self.body_pos.append(body_pos.copy())
        self.body_quat.append(body_quat.copy())
        self.body_lin_vel.append(body_lin_vel.copy())
        self.body_ang_vel.append(body_ang_vel.copy())
        self.foot_contact.append(foot_contact.copy())
        self.command.append(command.copy())
        self.timestamps.append(time.perf_counter())

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Stack lists → 2-D arrays; returns empty dict if no data."""
        if not self.timestamps:
            return {}
        return {
            "actions":      np.stack(self.actions),
            "joint_pos":    np.stack(self.joint_pos),
            "joint_vel":    np.stack(self.joint_vel),
            "body_pos":     np.stack(self.body_pos),
            "body_quat":    np.stack(self.body_quat),
            "body_lin_vel": np.stack(self.body_lin_vel),
            "body_ang_vel": np.stack(self.body_ang_vel),
            "foot_contact": np.stack(self.foot_contact),
            "command":      np.stack(self.command),
            "timestamps":   np.array(self.timestamps),
        }

    @property
    def n_steps(self) -> int:
        return len(self.timestamps)


# ── Live stats helpers ────────────────────────────────────────────────────────

def _quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """MuJoCo quat (w,x,y,z) → roll, pitch, yaw in degrees."""
    w, x, y, z = q
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return float(roll), float(pitch), float(yaw)


def _live_stats_line(rec: GaitRecorder) -> str:
    """Return a multi-line stats string from the last N steps."""
    n = min(rec.n_steps, 200)
    if n < 5:
        return "(waiting for data...)"
    act   = np.stack(rec.actions[-n:])          # (n, 12)
    jp    = np.stack(rec.joint_pos[-n:])         # (n, 12)
    fc    = np.stack(rec.foot_contact[-n:])      # (n, 4)
    bq    = np.stack(rec.body_quat[-n:])         # (n, 4)
    bav   = np.stack(rec.body_ang_vel[-n:])      # (n, 3)
    bp    = np.stack(rec.body_pos[-n:])          # (n, 3)
    cmd   = np.stack(rec.command[-n:])           # (n, 3)
    blv   = np.stack(rec.body_lin_vel[-n:])      # (n, 3)

    # Orientation stats
    euler = np.array([_quat_to_euler(bq[i]) for i in range(n)])
    roll_std  = float(np.std(euler[:, 0]))
    pitch_std = float(np.std(euler[:, 1]))
    roll_mean = float(np.mean(euler[:, 0]))
    pitch_mean= float(np.mean(euler[:, 1]))

    # Angular velocity
    ang_vel_mag = np.linalg.norm(bav, axis=1)
    ang_vel_std = float(np.std(ang_vel_mag))

    # Height
    height_mean = float(np.mean(bp[:, 2]))
    height_std  = float(np.std(bp[:, 2]))

    # Action smoothness: step-wise L2 change
    act_diff = np.diff(act, axis=0)
    act_rate = float(np.mean(np.linalg.norm(act_diff, axis=1)))

    # Action per-joint std (measure of oscillation)
    act_std = np.std(act, axis=0)  # (12,)

    # Contact ratio per foot
    contact_ratio = np.mean(fc > 0, axis=0)  # (4,)

    # Velocity tracking
    mean_cmd_vx  = float(np.mean(np.abs(cmd[:, 0])))
    mean_vel_vx  = float(np.mean(blv[:, 0]))
    mean_cmd_wz  = float(np.mean(np.abs(cmd[:, 2])))
    mean_vel_wz  = float(np.mean(np.abs(bav[:, 2])))

    lines = [
        f"  Steps recorded : {rec.n_steps:6d}",
        f"  Body height     : {height_mean:.3f} ± {height_std:.4f} m",
        f"  Roll  (mean±std): {roll_mean:+.2f}° ± {roll_std:.2f}°",
        f"  Pitch (mean±std): {pitch_mean:+.2f}° ± {pitch_std:.2f}°",
        f"  Ang vel magnitude std: {ang_vel_std:.4f} rad/s",
        f"  Action rate (L2/step): {act_rate:.4f}",
        f"  Contact ratio FL/FR/RL/RR: " +
        f"  {contact_ratio[1]:.2f}/{contact_ratio[0]:.2f}/"
        f"{contact_ratio[3]:.2f}/{contact_ratio[2]:.2f}",
        f"  Cmd vx={mean_cmd_vx:.2f}→act vx={mean_vel_vx:.2f} m/s | "
        f"Cmd wz={mean_cmd_wz:.2f}→act wz={mean_vel_wz:.2f} rad/s",
        f"  Joint action std (max): {float(np.max(act_std)):.4f}  "
        f"(min): {float(np.min(act_std)):.4f}",
    ]
    return "\n".join(lines)


# ── Keyboard command state ────────────────────────────────────────────────────

class _Ctrl:
    def __init__(self) -> None:
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

    def zero(self) -> None:
        self.vx = self.vy = self.wz = 0.0


# ── Gait monitoring viewer ────────────────────────────────────────────────────

class _GaitKeyboardViewer(NativeMujocoViewer):
    """Keyboard viewer that hooks into each env step to record gait data."""

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        policy,
        ctrl: _Ctrl,
        recorder: GaitRecorder,
        print_interval: int = 200,
    ) -> None:
        super().__init__(env, policy)
        self._ctrl = ctrl
        self._env_ref = env
        self._recorder = recorder
        self._print_interval = print_interval
        self._last_actions: Optional[np.ndarray] = None
        self._contact_sensor_name = "feet_ground_contact"
        self._has_contact_sensor = False  # resolved in setup()

    # Called after setup() resolves the env scene.
    def setup(self) -> None:
        super().setup()
        try:
            _ = self._env_ref.unwrapped.scene[self._contact_sensor_name]
            self._has_contact_sensor = True
        except (KeyError, AttributeError):
            self._has_contact_sensor = False

    def _inject_cmd(self) -> None:
        cmd = self._env_ref.unwrapped.command_manager.get_term("twist")
        cmd.vel_command_b[:, 0] = self._ctrl.vx
        cmd.vel_command_b[:, 1] = self._ctrl.vy
        cmd.vel_command_b[:, 2] = self._ctrl.wz
        cmd.time_left[:] = _FREEZE

    # Override _execute_step to intercept actions + state.
    def _execute_step(self) -> bool:
        try:
            with torch.no_grad():
                obs = self._env_ref.get_observations()
                actions = self.policy(obs)
                self._env_ref.step(actions)
                self._step_count += 1
                self._stats_steps += 1
                # Record gait data for env 0
                self._record_step(actions)
                return True
        except Exception:
            import traceback
            self._last_error = traceback.format_exc()
            self.pause()
            return False

    def _record_step(self, actions: torch.Tensor) -> None:
        """Extract and store state from env idx 0."""
        env_unwrapped = self._env_ref.unwrapped
        sim = env_unwrapped.sim
        sim_data = sim.data
        env_idx = 0

        # Joint positions/velocities (skip free joint: first 7/6 entries)
        qpos = sim_data.qpos[env_idx].cpu().numpy()   # (nq,)
        qvel = sim_data.qvel[env_idx].cpu().numpy()   # (nv,)

        body_pos  = qpos[0:3].copy()
        body_quat = qpos[3:7].copy()   # MuJoCo: w,x,y,z
        joint_pos = qpos[7:7+NUM_JOINTS].copy() if len(qpos) >= 7+NUM_JOINTS else np.zeros(NUM_JOINTS)

        body_lin_vel = qvel[0:3].copy()
        body_ang_vel = qvel[3:6].copy()
        joint_vel    = qvel[6:6+NUM_JOINTS].copy() if len(qvel) >= 6+NUM_JOINTS else np.zeros(NUM_JOINTS)

        # Foot contacts
        if self._has_contact_sensor:
            try:
                sensor = env_unwrapped.scene[self._contact_sensor_name]
                found  = sensor.data.found  # [B, N] or [B, N, 1]
                if found is not None:
                    fc = found[env_idx].cpu().numpy().flatten()[:NUM_FEET]
                    foot_contact = (fc > 0).astype(np.float32)
                else:
                    foot_contact = np.zeros(NUM_FEET, dtype=np.float32)
            except Exception:
                foot_contact = np.zeros(NUM_FEET, dtype=np.float32)
        else:
            foot_contact = np.zeros(NUM_FEET, dtype=np.float32)

        # Velocity command
        try:
            cmd_term = env_unwrapped.command_manager.get_term("twist")
            cmd = cmd_term.vel_command_b[env_idx, :3].cpu().numpy()
        except Exception:
            cmd = np.array([self._ctrl.vx, self._ctrl.vy, self._ctrl.wz], dtype=np.float32)

        # Actions (env 0)
        act_np = actions[env_idx].cpu().numpy()

        self._recorder.record(
            actions=act_np,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos=body_pos,
            body_quat=body_quat,
            body_lin_vel=body_lin_vel,
            body_ang_vel=body_ang_vel,
            foot_contact=foot_contact,
            command=cmd,
        )

        # Periodic live stats print (to stderr to avoid terminal flicker)
        if self._recorder.n_steps % self._print_interval == 0:
            stats = _live_stats_line(self._recorder)
            sys.stderr.write(f"\n\033[2m[GaitMonitor @ step {self._recorder.n_steps}]\033[0m\n{stats}\n")
            sys.stderr.flush()

    # ── Keyboard handling ─────────────────────────────────────────────────────

    def _handle_key(self, ch: str) -> None:
        c = self._ctrl
        changed = True
        if ch == "w":
            c.vx = min(c.vx + 0.3, _VX_MAX)
        elif ch == "s":
            c.vx = max(c.vx - 0.3, -_VX_MAX)
        elif ch == "a":
            c.vy = min(c.vy + 0.2, _VY_MAX)
        elif ch == "d":
            c.vy = max(c.vy - 0.2, -_VY_MAX)
        elif ch == "q":
            c.wz = min(c.wz + 0.3, _WZ_MAX)
        elif ch == "e":
            c.wz = max(c.wz - 0.3, -_WZ_MAX)
        elif ch == "x":
            c.zero()
        elif ch == "9":
            c.zero()
            self.request_reset()
        elif ch == "m":
            # Manual stats snapshot
            stats = _live_stats_line(self._recorder)
            sys.stderr.write(f"\n\033[1m[Manual Stats Snapshot]\033[0m\n{stats}\n")
            sys.stderr.flush()
            changed = False
        elif ch in ("\x03", "\x1b"):
            self.close()
            return
        else:
            changed = False

        if changed:
            _print_cmd(c, reset=(ch == "9"))

    def run(self) -> None:
        """Override run(): raw terminal + gait recording loop."""
        self.setup()
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setraw(fd)
        try:
            while self.is_running():
                while True:
                    r, _, _ = select.select([sys.stdin], [], [], 0)
                    if not r:
                        break
                    raw = os.read(fd, 1)
                    ch  = raw.decode("utf-8", errors="ignore").lower()
                    self._handle_key(ch)
                if not self.tick():
                    time.sleep(0.001)
                self._inject_cmd()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.close()


# ── Analysis & plotting ───────────────────────────────────────────────────────

def _analyse_and_plot(data: dict[str, np.ndarray], out_dir: Path, tag: str) -> None:
    """Compute gait metrics and save figures + text report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    actions      = data["actions"]       # (T, 12)
    joint_pos    = data["joint_pos"]     # (T, 12)
    joint_vel    = data["joint_vel"]     # (T, 12)
    body_pos     = data["body_pos"]      # (T, 3)
    body_quat    = data["body_quat"]     # (T, 4)
    body_lin_vel = data["body_lin_vel"]  # (T, 3)
    body_ang_vel = data["body_ang_vel"]  # (T, 3)
    foot_contact = data["foot_contact"]  # (T, 4)
    command      = data["command"]       # (T, 3)
    timestamps   = data["timestamps"]   # (T,)

    T = len(timestamps)
    t = timestamps - timestamps[0]  # seconds from start

    # ── Derived quantities ────────────────────────────────────────────────────
    # Euler angles (degrees)
    euler = np.array([_quat_to_euler(body_quat[i]) for i in range(T)])
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

    # Action rate (joint-wise L2 change per step)
    act_diff  = np.diff(actions, axis=0)           # (T-1, 12)
    act_rate  = np.linalg.norm(act_diff, axis=1)   # (T-1,)

    # Action smoothness per joint: std over a sliding window
    WIN = min(50, T // 4)
    act_std_ts = np.array([
        np.std(actions[max(0, i-WIN):i+1], axis=0) for i in range(T)
    ])  # (T, 12)

    # Body height
    height = body_pos[:, 2]

    # Foot contact fraction per foot
    contact_ratio = np.mean(foot_contact > 0, axis=0)  # (4,)

    # Gait diagram: detect stance/swing transitions
    # Stride period estimation via autocorrelation of contact signal
    def stride_period(contact_col: np.ndarray, dt: float) -> float:
        """Estimate stride period in seconds via autocorrelation."""
        x = contact_col.astype(float) - np.mean(contact_col)
        if np.std(x) < 1e-6:
            return float("nan")
        ac = np.correlate(x, x, mode="full")
        ac = ac[len(ac)//2:]
        ac /= ac[0]
        # Find first peak after lag=1
        peaks = []
        for i in range(2, min(len(ac)-1, 500)):
            if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0.1:
                peaks.append(i)
                break
        return peaks[0] * dt if peaks else float("nan")

    # Approximate dt from step_dt (50 Hz control = 0.02s per step)
    dt = float(np.mean(np.diff(t))) if T > 1 else 0.02

    periods = [stride_period(foot_contact[:, i], dt) for i in range(NUM_FEET)]

    # Velocity tracking error
    vel_err_xy = np.sqrt((body_lin_vel[:, 0] - command[:, 0])**2 +
                         (body_lin_vel[:, 1] - command[:, 1])**2)
    yaw_err    = np.abs(body_ang_vel[:, 2] - command[:, 2])

    # Left-right symmetry: compare FL/FR and RL/RR joint actions
    # Actuator order: FR=0:3, FL=3:6, RR=6:9, RL=9:12
    fl_act = actions[:, 3:6]   # FL hip/thigh/calf
    fr_act = actions[:, 0:3]   # FR hip/thigh/calf
    rl_act = actions[:, 9:12]  # RL hip/thigh/calf
    rr_act = actions[:, 6:9]   # RR hip/thigh/calf

    # Hip symmetry: FL_hip vs -FR_hip (mirror)
    front_sym = np.mean(np.abs(fl_act[:, 0] + fr_act[:, 0]))   # hip (mirror)
    front_sym_thigh = np.mean(np.abs(fl_act[:, 1] - fr_act[:, 1]))  # thigh (same)
    rear_sym  = np.mean(np.abs(rl_act[:, 0] + rr_act[:, 0]))
    rear_sym_thigh  = np.mean(np.abs(rl_act[:, 1] - rr_act[:, 1]))

    # ── Build figures ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 26))
    fig.suptitle(
        f"Gait Stability Analysis  |  {tag}\n"
        f"Total steps: {T}  |  Duration: {t[-1]:.1f}s  |  "
        f"Step dt: {dt*1000:.1f}ms",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: Body orientation ───────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(t, roll,  label=f"Roll  (std={np.std(roll):.2f}°)",  alpha=0.8)
    ax0.plot(t, pitch, label=f"Pitch (std={np.std(pitch):.2f}°)", alpha=0.8)
    ax0.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax0.set_title("Body Orientation (Roll & Pitch)")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Angle (°)")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax0b = fig.add_subplot(gs[0, 2])
    ax0b.bar(["Roll std", "Pitch std"], [np.std(roll), np.std(pitch)],
             color=["steelblue", "coral"])
    ax0b.set_title("Orientation Stability")
    ax0b.set_ylabel("Std dev (°)")
    ax0b.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate([np.std(roll), np.std(pitch)]):
        ax0b.text(i, v + 0.05, f"{v:.2f}°", ha="center", fontsize=9)

    # ── Row 1: Body height & angular velocity ─────────────────────────────────
    ax1 = fig.add_subplot(gs[1, :2])
    ax1.plot(t, height, color="green", label=f"Height (mean={np.mean(height):.3f}m, std={np.std(height):.4f}m)", alpha=0.8)
    ax1.axhline(0.3, color="red", linewidth=1, linestyle="--", label="Target 0.30m")
    ax1.set_title("Body Height")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Height (m)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ang_vel_mag = np.linalg.norm(body_ang_vel, axis=1)
    ax1b = fig.add_subplot(gs[1, 2])
    ax1b.plot(t, ang_vel_mag, color="purple", alpha=0.7, linewidth=0.8)
    ax1b.set_title(f"Body Ang. Vel. (std={np.std(ang_vel_mag):.4f})")
    ax1b.set_xlabel("Time (s)")
    ax1b.set_ylabel("||ω|| (rad/s)")
    ax1b.grid(True, alpha=0.3)

    # ── Row 2: Action rate & smoothness ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, :2])
    ax2.plot(t[1:], act_rate, color="darkorange", alpha=0.7, linewidth=0.8,
             label=f"Action L2 rate (mean={np.mean(act_rate):.4f})")
    ax2.set_title("Action Rate (L2 Norm of Per-Step Change)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("||Δaction||")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    act_std_mean = np.mean(act_std_ts, axis=0)   # (12,)
    ax2b = fig.add_subplot(gs[2, 2])
    bars = ax2b.bar(range(NUM_JOINTS), act_std_mean, color="darkorange", alpha=0.8)
    ax2b.set_xticks(range(NUM_JOINTS))
    ax2b.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=6)
    ax2b.set_title("Per-Joint Action Std Dev")
    ax2b.set_ylabel("Std Dev")
    ax2b.grid(True, alpha=0.3, axis="y")

    # ── Row 3: Foot contact gait diagram ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[3, :])
    # Show at most 10 seconds
    t_mask = t <= min(t[-1], 10.0)
    t_short = t[t_mask]
    fc_short = foot_contact[t_mask]
    for fi, fname in enumerate(FOOT_NAMES):
        y_base = fi
        contact_f = fc_short[:, fi]
        # Fill regions
        in_stance = False
        start_t = 0.0
        for ti, (ti_val, ci) in enumerate(zip(t_short, contact_f)):
            if ci > 0 and not in_stance:
                start_t = ti_val
                in_stance = True
            elif ci == 0 and in_stance:
                ax3.barh(y_base, ti_val - start_t, left=start_t,
                         height=0.6, align="center", color=f"C{fi}", alpha=0.7)
                in_stance = False
        if in_stance:
            ax3.barh(y_base, t_short[-1] - start_t, left=start_t,
                     height=0.6, align="center", color=f"C{fi}", alpha=0.7,
                     label=f"{fname} (stance {contact_ratio[fi]*100:.0f}%)")
        else:
            ax3.barh(y_base, 0, height=0.6, color=f"C{fi}", alpha=0.7,
                     label=f"{fname} (stance {contact_ratio[fi]*100:.0f}%)")
    ax3.set_yticks(range(NUM_FEET))
    ax3.set_yticklabels(FOOT_NAMES, fontsize=9)
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Gait Diagram (first 10s, filled = stance)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.2, axis="x")
    # Annotate stride periods
    for fi, (fname, per) in enumerate(zip(FOOT_NAMES, periods)):
        if not np.isnan(per):
            ax3.text(t_short[-1] * 0.02, fi, f" T={per*1000:.0f}ms",
                     va="center", fontsize=7, color="black")

    # ── Row 4: Velocity tracking ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[4, :2])
    ax4.plot(t, command[:, 0],      "--", label="Cmd vx", color="blue",   alpha=0.7)
    ax4.plot(t, body_lin_vel[:, 0],       label="Act vx", color="navy",   alpha=0.7)
    ax4.plot(t, command[:, 2],      "--", label="Cmd wz", color="red",    alpha=0.7)
    ax4.plot(t, body_ang_vel[:, 2],       label="Act wz", color="darkred",alpha=0.7)
    ax4.set_title("Velocity Tracking (command vs actual)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Velocity")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Velocity tracking error
    ax4b = fig.add_subplot(gs[4, 2])
    ax4b.plot(t, vel_err_xy, color="blue",   alpha=0.7, linewidth=0.8, label=f"xy err (mean={np.mean(vel_err_xy):.3f})")
    ax4b.plot(t, yaw_err,    color="red",    alpha=0.7, linewidth=0.8, label=f"yaw err (mean={np.mean(yaw_err):.3f})")
    ax4b.set_title("Velocity Tracking Error")
    ax4b.set_xlabel("Time (s)")
    ax4b.set_ylabel("Error")
    ax4b.legend(fontsize=8)
    ax4b.grid(True, alpha=0.3)

    # ── Row 5: L-R symmetry & joint actions per leg ───────────────────────────
    ax5 = fig.add_subplot(gs[5, :2])
    for ji in range(NUM_JOINTS):
        ax5.plot(t, actions[:, ji], alpha=0.5, linewidth=0.6, label=JOINT_NAMES[ji])
    ax5.set_title("All Joint Actions Over Time")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Action")
    ax5.legend(loc="upper right", fontsize=6, ncol=3)
    ax5.grid(True, alpha=0.2)

    # Symmetry bar chart
    sym_labels = ["F-hip\n(mirror)", "F-thigh\n(same)", "R-hip\n(mirror)", "R-thigh\n(same)"]
    sym_vals   = [front_sym, front_sym_thigh, rear_sym, rear_sym_thigh]
    ax5b = fig.add_subplot(gs[5, 2])
    ax5b.bar(sym_labels, sym_vals, color=["steelblue", "coral", "steelblue", "coral"])
    ax5b.set_title("L-R Symmetry Error\n(lower = more symmetric)")
    ax5b.set_ylabel("Mean |L - mirror(R)|")
    ax5b.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(sym_vals):
        ax5b.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"gait_report_{tag}.png"
    fig.savefig(str(plot_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[GaitMonitor] Plot saved → {plot_path}")

    # ── Text summary ──────────────────────────────────────────────────────────
    txt_path = out_dir / f"gait_summary_{tag}.txt"
    with open(txt_path, "w") as f:
        def w(s=""): f.write(s + "\n")
        w("=" * 70)
        w(f"  GAIT STABILITY ANALYSIS — {tag}")
        w("=" * 70)
        w(f"  Total steps         : {T}")
        w(f"  Duration            : {t[-1]:.2f} s")
        w(f"  Step dt             : {dt*1000:.2f} ms")
        w()
        w("── Body Orientation ──────────────────────────────────────────────")
        w(f"  Roll   mean ± std   : {np.mean(roll):+.2f}° ± {np.std(roll):.2f}°")
        w(f"  Pitch  mean ± std   : {np.mean(pitch):+.2f}° ± {np.std(pitch):.2f}°")
        w(f"  Roll   max abs      : {np.max(np.abs(roll)):.2f}°")
        w(f"  Pitch  max abs      : {np.max(np.abs(pitch)):.2f}°")
        w()
        w("── Body Height ───────────────────────────────────────────────────")
        w(f"  Height mean ± std   : {np.mean(height):.4f} ± {np.std(height):.4f} m")
        w(f"  Height min / max    : {np.min(height):.4f} / {np.max(height):.4f} m")
        w(f"  Deviation from 0.3m : {abs(np.mean(height)-0.3):.4f} m")
        w()
        w("── Angular Velocity ──────────────────────────────────────────────")
        w(f"  ||ω||  mean ± std   : {np.mean(ang_vel_mag):.4f} ± {np.std(ang_vel_mag):.4f} rad/s")
        w(f"  ||ω||  max          : {np.max(ang_vel_mag):.4f} rad/s")
        w()
        w("── Action Smoothness ─────────────────────────────────────────────")
        w(f"  Action rate mean    : {np.mean(act_rate):.4f}")
        w(f"  Action rate std     : {np.std(act_rate):.4f}")
        w(f"  Action rate max     : {np.max(act_rate):.4f}")
        w()
        w("── Per-Joint Action Stats ────────────────────────────────────────")
        w(f"  {'Joint':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        for ji, jn in enumerate(JOINT_NAMES):
            a = actions[:, ji]
            w(f"  {jn:<15} {np.mean(a):>8.4f} {np.std(a):>8.4f} {np.min(a):>8.4f} {np.max(a):>8.4f}")
        w()
        w("── Foot Contact ──────────────────────────────────────────────────")
        for fi, fn in enumerate(FOOT_NAMES):
            per_str = f"{periods[fi]*1000:.0f} ms" if not np.isnan(periods[fi]) else "N/A"
            w(f"  {fn}  stance ratio: {contact_ratio[fi]*100:.1f}%  |  stride period ≈ {per_str}")
        w()
        w("── Velocity Tracking ─────────────────────────────────────────────")
        w(f"  xy vel error mean   : {np.mean(vel_err_xy):.4f} m/s")
        w(f"  yaw vel error mean  : {np.mean(yaw_err):.4f} rad/s")
        w()
        w("── L-R Symmetry ──────────────────────────────────────────────────")
        w(f"  Front hip  (mirror) : {front_sym:.4f}")
        w(f"  Front thigh (same)  : {front_sym_thigh:.4f}")
        w(f"  Rear  hip  (mirror) : {rear_sym:.4f}")
        w(f"  Rear  thigh (same)  : {rear_sym_thigh:.4f}")
        w("=" * 70)
    print(f"[GaitMonitor] Summary saved → {txt_path}")

    # Print summary to terminal
    with open(txt_path) as f:
        print(f.read())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_cmd(c: _Ctrl, reset: bool = False) -> None:
    tag = "RESET" if reset else "CMD  "
    sys.stdout.write(
        f"\r[{tag}] vx={c.vx:+.2f}  vy={c.vy:+.2f}  wz={c.wz:+.2f}"
        "    w/s=前后  a/d=左右  q/e=转向  x=停  m=步态快照  9=重置  ESC=退出  "
    )
    sys.stdout.flush()


def _print_help() -> None:
    print(
        "\n"
        "┌──────────────────────────────────────────────────────────────┐\n"
        "│       Terminal Keyboard + Gait Monitor                       │\n"
        "│  w / s  : 前进 / 后退       +/- 0.30 m/s                   │\n"
        "│  a / d  : 左平移 / 右平移   +/- 0.20 m/s                   │\n"
        "│  q / e  : 左转  / 右转      +/- 0.30 rad/s                 │\n"
        "│  x      : 速度清零（停止原地）                             │\n"
        "│  m      : 打印当前步态统计快照                             │\n"
        "│  9      : 重置环境 + 速度清零                              │\n"
        "│  ESC / Ctrl-C : 退出 + 生成分析报告                       │\n"
        "│                                                              │\n"
        "│  步态统计每 200 步自动输出，退出时保存图表和文本报告       │\n"
        "└──────────────────────────────────────────────────────────────┘"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Cfg:
    checkpoint_file: str
    """Path to model checkpoint (.pt)."""
    num_envs: int = 1
    """Number of parallel environments."""
    device: str | None = None
    """Torch device (default: cuda:0 if available, else cpu)."""
    print_interval: int = 200
    """Print live stats every N steps."""
    out_dir: str = "logs/gait_monitor"
    """Directory for analysis outputs."""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import mjlab.tasks  # noqa: F401
    import src.tasks    # noqa: F401

    from mjlab.tasks.registry import list_tasks

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    cfg = tyro.cli(
        _Cfg,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg    = load_env_cfg(chosen_task, play=True)
    agent_cfg  = load_rl_cfg(chosen_task)
    env_cfg.scene.num_envs = cfg.num_envs

    if env_cfg.scene.terrain is not None:
        env_cfg.scene.terrain.terrain_generator = _KEYBOARD_TERRAIN_CFG

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    twist_cmd = env_cfg.commands.get("twist")
    if isinstance(twist_cmd, UniformVelocityCommandCfg):
        twist_cmd.heading_command = False
        twist_cmd.ranges.heading = None

    env_cfg.terminations = {}

    env    = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env    = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    print(f"[INFO] Loading: {resume_path.name}")

    runner_cls = load_runner_cls(chosen_task) or MjlabOnPolicyRunner
    runner     = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    ctrl     = _Ctrl()
    recorder = GaitRecorder()

    _print_help()

    viewer = _GaitKeyboardViewer(
        env, policy, ctrl, recorder,
        print_interval=cfg.print_interval,
    )
    viewer.run()

    # ── Post-run analysis ─────────────────────────────────────────────────────
    data = recorder.to_arrays()
    if data:
        tag     = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg.out_dir)
        # Save raw data
        npz_path = out_dir / f"gait_data_{tag}.npz"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(str(npz_path), **data)
        print(f"\n[GaitMonitor] Raw data saved → {npz_path}")
        print(f"[GaitMonitor] Running analysis on {len(data['timestamps'])} steps...")
        _analyse_and_plot(data, out_dir, tag)
    else:
        print("\n[GaitMonitor] No data recorded.")

    env.close()


if __name__ == "__main__":
    main()
