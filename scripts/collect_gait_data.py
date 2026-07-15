"""Headless gait data collection script.

Runs the RL policy for a specified number of steps without any viewer,
collects per-step state data, and generates a gait stability analysis report.

Usage:
    python scripts/collect_gait_data.py Boying-HIM-Rough \
        --checkpoint_file=logs/rsl_rl/boying_him_velocity/2026-07-13_19-51-48/model_49999.pt \
        --num_steps=2000 \
        --vx=1.0  # forward velocity command

Output (logs/gait_monitor/):
    gait_data_<timestamp>.npz
    gait_report_<timestamp>.png
    gait_summary_<timestamp>.txt
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tyro

JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]
FOOT_NAMES = ["FR", "FL", "RR", "RL"]
NUM_JOINTS = 12
NUM_FEET   = 4


def _quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """MuJoCo quat (w,x,y,z) → roll, pitch, yaw in degrees."""
    w, x, y, z = q
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return float(roll), float(pitch), float(yaw)


def collect_data(
    env,
    policy,
    num_steps: int,
    vx: float,
    vy: float,
    wz: float,
    freeze_cmd: bool = True,
) -> dict[str, np.ndarray]:
    """Run policy for num_steps and return collected arrays."""
    FREEZE = 1e6

    actions_list:      list[np.ndarray] = []
    joint_pos_list:    list[np.ndarray] = []
    joint_vel_list:    list[np.ndarray] = []
    body_pos_list:     list[np.ndarray] = []
    body_quat_list:    list[np.ndarray] = []
    body_lin_vel_list: list[np.ndarray] = []
    body_ang_vel_list: list[np.ndarray] = []
    foot_contact_list: list[np.ndarray] = []
    command_list:      list[np.ndarray] = []

    # Check for contact sensor
    has_contact = False
    try:
        _ = env.unwrapped.scene["feet_ground_contact"]
        has_contact = True
    except (KeyError, AttributeError):
        pass

    print(f"[INFO] Contact sensor available: {has_contact}")
    print(f"[INFO] Collecting {num_steps} steps with cmd vx={vx:.2f} vy={vy:.2f} wz={wz:.2f}...")

    for step in range(num_steps):
        # Inject fixed velocity command
        if freeze_cmd:
            try:
                cmd_term = env.unwrapped.command_manager.get_term("twist")
                cmd_term.vel_command_b[:, 0] = vx
                cmd_term.vel_command_b[:, 1] = vy
                cmd_term.vel_command_b[:, 2] = wz
                cmd_term.time_left[:] = FREEZE
            except Exception:
                pass

        # Policy step
        with torch.no_grad():
            obs = env.get_observations()
            actions = policy(obs)
            env.step(actions)

        # Extract data (env idx 0)
        sim_data = env.unwrapped.sim.data
        qpos = sim_data.qpos[0].cpu().numpy()
        qvel = sim_data.qvel[0].cpu().numpy()

        body_pos  = qpos[0:3].copy()
        body_quat = qpos[3:7].copy()
        joint_pos = qpos[7:7+NUM_JOINTS].copy() if len(qpos) >= 7+NUM_JOINTS else np.zeros(NUM_JOINTS)
        body_lin_vel = qvel[0:3].copy()
        body_ang_vel = qvel[3:6].copy()
        joint_vel    = qvel[6:6+NUM_JOINTS].copy() if len(qvel) >= 6+NUM_JOINTS else np.zeros(NUM_JOINTS)

        if has_contact:
            try:
                sensor = env.unwrapped.scene["feet_ground_contact"]
                found  = sensor.data.found
                if found is not None:
                    fc = found[0].cpu().numpy().flatten()[:NUM_FEET]
                    foot_contact = (fc > 0).astype(np.float32)
                else:
                    foot_contact = np.zeros(NUM_FEET, dtype=np.float32)
            except Exception:
                foot_contact = np.zeros(NUM_FEET, dtype=np.float32)
        else:
            foot_contact = np.zeros(NUM_FEET, dtype=np.float32)

        try:
            cmd_term = env.unwrapped.command_manager.get_term("twist")
            cmd = cmd_term.vel_command_b[0, :3].cpu().numpy()
        except Exception:
            cmd = np.array([vx, vy, wz], dtype=np.float32)

        act_np = actions[0].cpu().numpy()

        actions_list.append(act_np.copy())
        joint_pos_list.append(joint_pos)
        joint_vel_list.append(joint_vel)
        body_pos_list.append(body_pos)
        body_quat_list.append(body_quat)
        body_lin_vel_list.append(body_lin_vel)
        body_ang_vel_list.append(body_ang_vel)
        foot_contact_list.append(foot_contact)
        command_list.append(cmd.copy())

        if (step + 1) % 500 == 0:
            print(f"  step {step+1}/{num_steps}  height={body_pos[2]:.3f}m  "
                  f"vx_act={body_lin_vel[0]:.2f}m/s")

    return {
        "actions":      np.stack(actions_list),
        "joint_pos":    np.stack(joint_pos_list),
        "joint_vel":    np.stack(joint_vel_list),
        "body_pos":     np.stack(body_pos_list),
        "body_quat":    np.stack(body_quat_list),
        "body_lin_vel": np.stack(body_lin_vel_list),
        "body_ang_vel": np.stack(body_ang_vel_list),
        "foot_contact": np.stack(foot_contact_list),
        "command":      np.stack(command_list),
    }


def analyse_and_plot(data: dict[str, np.ndarray], out_dir: Path, tag: str,
                     vx: float, vy: float, wz: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    actions      = data["actions"]
    joint_pos    = data["joint_pos"]
    body_pos     = data["body_pos"]
    body_quat    = data["body_quat"]
    body_lin_vel = data["body_lin_vel"]
    body_ang_vel = data["body_ang_vel"]
    foot_contact = data["foot_contact"]
    command      = data["command"]

    T  = len(actions)
    dt = 0.02   # env step_dt = 0.02s
    t  = np.arange(T) * dt

    euler = np.array([_quat_to_euler(body_quat[i]) for i in range(T)])
    roll, pitch = euler[:, 0], euler[:, 1]
    height = body_pos[:, 2]
    ang_vel_mag = np.linalg.norm(body_ang_vel, axis=1)

    act_diff  = np.diff(actions, axis=0)
    act_rate  = np.linalg.norm(act_diff, axis=1)

    contact_ratio = np.mean(foot_contact > 0, axis=0)
    vel_err_xy = np.sqrt((body_lin_vel[:, 0] - command[:, 0])**2 +
                         (body_lin_vel[:, 1] - command[:, 1])**2)
    yaw_err    = np.abs(body_ang_vel[:, 2] - command[:, 2])

    def stride_period(contact_col: np.ndarray, dt: float) -> float:
        x = contact_col.astype(float) - np.mean(contact_col)
        if np.std(x) < 1e-6:
            return float("nan")
        ac = np.correlate(x, x, mode="full")
        ac = ac[len(ac)//2:]
        ac /= ac[0]
        for i in range(2, min(len(ac)-1, 500)):
            if ac[i] > ac[i-1] and ac[i] > ac[i+1] and ac[i] > 0.1:
                return i * dt
        return float("nan")

    periods = [stride_period(foot_contact[:, i], dt) for i in range(NUM_FEET)]

    fl_act = actions[:, 3:6]
    fr_act = actions[:, 0:3]
    rl_act = actions[:, 9:12]
    rr_act = actions[:, 6:9]
    front_sym       = float(np.mean(np.abs(fl_act[:, 0] + fr_act[:, 0])))
    front_sym_thigh = float(np.mean(np.abs(fl_act[:, 1] - fr_act[:, 1])))
    rear_sym        = float(np.mean(np.abs(rl_act[:, 0] + rr_act[:, 0])))
    rear_sym_thigh  = float(np.mean(np.abs(rl_act[:, 1] - rr_act[:, 1])))

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 28))
    fig.suptitle(
        f"步态稳定性分析  |  {tag}\n"
        f"Steps: {T}  |  Duration: {t[-1]:.1f}s  |  "
        f"Command: vx={vx:.1f} vy={vy:.1f} wz={wz:.1f}",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(7, 3, figure=fig, hspace=0.5, wspace=0.35)

    # Row 0: orientation
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(t, roll,  label=f"Roll  μ={np.mean(roll):+.2f}° σ={np.std(roll):.2f}°",  alpha=0.8)
    ax.plot(t, pitch, label=f"Pitch μ={np.mean(pitch):+.2f}° σ={np.std(pitch):.2f}°", alpha=0.8)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set(title="机体姿态角（Roll & Pitch）", xlabel="Time (s)", ylabel="Angle (°)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(["Roll σ", "Pitch σ"], [np.std(roll), np.std(pitch)], color=["steelblue","coral"])
    ax.set(title="姿态稳定性（σ越小越稳）", ylabel="Std dev (°)")
    for i, v in enumerate([np.std(roll), np.std(pitch)]):
        ax.text(i, v+0.02, f"{v:.2f}°", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Row 1: height
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(t, height, color="green",
            label=f"Height μ={np.mean(height):.3f}m σ={np.std(height):.4f}m", alpha=0.8)
    ax.axhline(0.3, color="red", lw=1, ls="--", label="Target 0.30m")
    ax.set(title="机体高度", xlabel="Time (s)", ylabel="Height (m)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, ang_vel_mag, color="purple", alpha=0.7, lw=0.8)
    ax.set(title=f"机体角速度大小\nμ={np.mean(ang_vel_mag):.3f} σ={np.std(ang_vel_mag):.4f} rad/s",
           xlabel="Time (s)", ylabel="||ω|| (rad/s)")
    ax.grid(alpha=0.3)

    # Row 2: action rate
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(t[1:], act_rate, color="darkorange", alpha=0.7, lw=0.8,
            label=f"Action rate μ={np.mean(act_rate):.4f} σ={np.std(act_rate):.4f}")
    ax.set(title="动作变化率（平滑性指标）", xlabel="Time (s)", ylabel="||Δaction||₂")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    act_std = np.std(actions, axis=0)
    ax = fig.add_subplot(gs[2, 2])
    ax.bar(range(NUM_JOINTS), act_std, color="darkorange", alpha=0.8)
    ax.set_xticks(range(NUM_JOINTS)); ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=6)
    ax.set(title="各关节动作标准差", ylabel="Std Dev")
    ax.grid(alpha=0.3, axis="y")

    # Row 3: gait diagram
    ax = fig.add_subplot(gs[3, :])
    t_lim = min(t[-1], 10.0)
    t_mask = t <= t_lim
    t_s, fc_s = t[t_mask], foot_contact[t_mask]
    for fi, fn in enumerate(FOOT_NAMES):
        in_stance, start_t = False, 0.0
        for ti_val, ci in zip(t_s, fc_s[:, fi]):
            if ci > 0 and not in_stance:
                start_t = ti_val; in_stance = True
            elif ci == 0 and in_stance:
                ax.barh(fi, ti_val-start_t, left=start_t, height=0.6, color=f"C{fi}", alpha=0.75)
                in_stance = False
        if in_stance:
            ax.barh(fi, t_s[-1]-start_t, left=start_t, height=0.6,
                    color=f"C{fi}", alpha=0.75,
                    label=f"{fn} 支撑{contact_ratio[fi]*100:.0f}%")
        else:
            ax.barh(fi, 0, height=0.6, color=f"C{fi}", alpha=0.75,
                    label=f"{fn} 支撑{contact_ratio[fi]*100:.0f}%")
        per_str = f"T≈{periods[fi]*1000:.0f}ms" if not np.isnan(periods[fi]) else ""
        ax.text(0.02*t_lim, fi, f" {per_str}", va="center", fontsize=7)
    ax.set_yticks(range(NUM_FEET)); ax.set_yticklabels(FOOT_NAMES, fontsize=9)
    ax.set(title="步态图（前10秒，填充=支撑相）", xlabel="Time (s)")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.2, axis="x")

    # Row 4: velocity tracking
    ax = fig.add_subplot(gs[4, :2])
    ax.plot(t, command[:, 0], "--", label=f"Cmd vx={vx:.1f}", color="blue", alpha=0.7)
    ax.plot(t, body_lin_vel[:, 0], label="Act vx", color="navy", alpha=0.8)
    ax.plot(t, command[:, 2], "--", label=f"Cmd wz={wz:.1f}", color="red", alpha=0.7)
    ax.plot(t, body_ang_vel[:, 2], label="Act wz", color="darkred", alpha=0.8)
    ax.set(title="速度跟踪（指令 vs 实际）", xlabel="Time (s)", ylabel="Velocity")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[4, 2])
    ax.plot(t, vel_err_xy, color="blue", alpha=0.7, lw=0.8,
            label=f"xy err μ={np.mean(vel_err_xy):.3f}")
    ax.plot(t, yaw_err, color="red", alpha=0.7, lw=0.8,
            label=f"yaw err μ={np.mean(yaw_err):.3f}")
    ax.set(title="速度跟踪误差", xlabel="Time (s)", ylabel="Error")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Row 5: joint actions per leg (FR/FL/RR/RL each in a mini-plot)
    leg_slices = {"FR": slice(0,3), "FL": slice(3,6), "RR": slice(6,9), "RL": slice(9,12)}
    for col, (leg, sl) in enumerate(leg_slices.items()):
        ax = fig.add_subplot(gs[5, col] if col < 3 else gs[6, 0])
        for ji, jname in zip(range(sl.start, sl.stop), JOINT_NAMES[sl]):
            suffix = jname.split("_")[1]
            ax.plot(t, actions[:, ji], alpha=0.8, lw=0.8, label=suffix)
        ax.set(title=f"{leg} 腿关节动作", xlabel="Time (s)", ylabel="Action")
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # Row 6: symmetry + phase diff
    ax = fig.add_subplot(gs[6, 0])
    sym_labels = ["F-hip\n(mirror)", "F-thigh\n(same)", "R-hip\n(mirror)", "R-thigh\n(same)"]
    ax.bar(sym_labels, [front_sym, front_sym_thigh, rear_sym, rear_sym_thigh],
           color=["steelblue","coral","steelblue","coral"])
    ax.set(title="左右对称性误差\n（越低越对称）", ylabel="Mean |L - mirror(R)|")
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[6, 1:])
    ax.plot(t, body_lin_vel[:, 0], label="vx actual", alpha=0.8)
    ax.plot(t, body_lin_vel[:, 1], label="vy actual", alpha=0.8)
    ax.plot(t, height, label="height", alpha=0.8, color="green")
    ax.set(title="综合状态时序", xlabel="Time (s)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"gait_report_{tag}.png"
    fig.savefig(str(plot_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[GaitMonitor] 图表已保存 → {plot_path}")

    # ── Text report ──────────────────────────────────────────────────────────
    txt = []
    def w(s=""): txt.append(s)
    w("=" * 70)
    w(f"  步态稳定性分析报告  {tag}")
    w(f"  命令速度: vx={vx:.2f} vy={vy:.2f} wz={wz:.2f}")
    w(f"  总步数: {T}  时长: {t[-1]:.2f}s")
    w("=" * 70)
    w()
    w("── 机体姿态 ──────────────────────────────────────────────────────────")
    w(f"  Roll  均值±标准差 : {np.mean(roll):+.2f}° ± {np.std(roll):.2f}°  最大: {np.max(np.abs(roll)):.2f}°")
    w(f"  Pitch 均值±标准差 : {np.mean(pitch):+.2f}° ± {np.std(pitch):.2f}°  最大: {np.max(np.abs(pitch)):.2f}°")
    w()
    w("── 机体高度 ──────────────────────────────────────────────────────────")
    w(f"  均值±标准差 : {np.mean(height):.4f} ± {np.std(height):.4f} m")
    w(f"  min/max     : {np.min(height):.4f} / {np.max(height):.4f} m")
    w(f"  偏差(目标0.3m): {abs(np.mean(height)-0.3):.4f} m")
    w()
    w("── 角速度 ────────────────────────────────────────────────────────────")
    w(f"  ||ω|| 均值±标准差 : {np.mean(ang_vel_mag):.4f} ± {np.std(ang_vel_mag):.4f} rad/s")
    w(f"  ||ω|| 最大         : {np.max(ang_vel_mag):.4f} rad/s")
    w()
    w("── 动作平滑性 ────────────────────────────────────────────────────────")
    w(f"  动作变化率 均值 : {np.mean(act_rate):.4f}")
    w(f"  动作变化率 标准差 : {np.std(act_rate):.4f}")
    w(f"  动作变化率 最大值 : {np.max(act_rate):.4f}")
    w()
    w("── 各关节动作统计 ────────────────────────────────────────────────────")
    w(f"  {'关节':<15} {'均值':>8} {'标准差':>8} {'最小':>8} {'最大':>8}")
    for ji, jn in enumerate(JOINT_NAMES):
        a = actions[:, ji]
        w(f"  {jn:<15} {np.mean(a):>8.4f} {np.std(a):>8.4f} {np.min(a):>8.4f} {np.max(a):>8.4f}")
    w()
    w("── 足端接触 ──────────────────────────────────────────────────────────")
    for fi, fn in enumerate(FOOT_NAMES):
        per_str = f"{periods[fi]*1000:.0f} ms" if not np.isnan(periods[fi]) else "N/A"
        w(f"  {fn}  支撑比: {contact_ratio[fi]*100:.1f}%  步态周期≈ {per_str}")
    w()
    w("── 速度跟踪 ──────────────────────────────────────────────────────────")
    w(f"  xy速度误差均值  : {np.mean(vel_err_xy):.4f} m/s")
    w(f"  偏航速度误差均值: {np.mean(yaw_err):.4f} rad/s")
    w()
    w("── 左右对称性 ────────────────────────────────────────────────────────")
    w(f"  前腿髋关节(镜像): {front_sym:.4f}")
    w(f"  前腿大腿(同向)  : {front_sym_thigh:.4f}")
    w(f"  后腿髋关节(镜像): {rear_sym:.4f}")
    w(f"  后腿大腿(同向)  : {rear_sym_thigh:.4f}")
    w("=" * 70)

    report = "\n".join(txt)
    print(report)

    txt_path = out_dir / f"gait_summary_{tag}.txt"
    txt_path.write_text(report, encoding="utf-8")
    print(f"[GaitMonitor] 文本报告已保存 → {txt_path}")



@dataclass(frozen=True)
class _Cfg:
    checkpoint_file: str
    """Path to model checkpoint (.pt)."""
    num_steps: int = 2000
    """Number of steps to collect."""
    vx: float = 1.0
    """Forward velocity command (m/s)."""
    vy: float = 0.0
    """Lateral velocity command (m/s)."""
    wz: float = 0.0
    """Yaw velocity command (rad/s)."""
    device: str | None = None
    out_dir: str = "logs/gait_monitor"


def main() -> None:
    import mjlab
    import mjlab.tasks  # noqa: F401
    import src.tasks    # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

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

    env_cfg   = load_env_cfg(chosen_task, play=True)
    agent_cfg = load_rl_cfg(chosen_task)
    env_cfg.scene.num_envs = 1

    # Use flat terrain for consistent gait analysis
    from mjlab.terrains import BoxFlatTerrainCfg, TerrainGeneratorCfg
    env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=10.0,
        num_rows=2,
        num_cols=4,
        curriculum=False,
        sub_terrains={"flat": BoxFlatTerrainCfg(proportion=1.0)},
    )

    # Disable terminations
    env_cfg.terminations = {}

    # Fix velocity command range to exactly cfg.vx/vy/wz
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    twist_cmd = env_cfg.commands.get("twist")
    if isinstance(twist_cmd, UniformVelocityCommandCfg):
        twist_cmd.heading_command = False
        twist_cmd.ranges.heading = None

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    print(f"[INFO] Loading checkpoint: {resume_path.name}")

    runner_cls = load_runner_cls(chosen_task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Warm up (let policy stabilize)
    print("[INFO] Warming up for 200 steps...")
    for _ in range(200):
        with torch.no_grad():
            obs = env.get_observations()
            act = policy(obs)
            env.step(act)
        try:
            cmd_term = env.unwrapped.command_manager.get_term("twist")
            cmd_term.vel_command_b[:, 0] = cfg.vx
            cmd_term.vel_command_b[:, 1] = cfg.vy
            cmd_term.vel_command_b[:, 2] = cfg.wz
            cmd_term.time_left[:] = 1e6
        except Exception:
            pass

    # Collect
    data = collect_data(env, policy, cfg.num_steps, cfg.vx, cfg.vy, cfg.wz)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"gait_data_{tag}.npz"
    np.savez(str(npz_path), **data)
    print(f"\n[GaitMonitor] 原始数据已保存 → {npz_path}")

    analyse_and_plot(data, out_dir, tag, cfg.vx, cfg.vy, cfg.wz)
    env.close()


if __name__ == "__main__":
    main()
