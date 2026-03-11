"""
Step 7: Evaluate a Trained Policy
===================================
Runs a trained policy in the OpenCabinet environment and reports
success rate across multiple episodes and kitchen scenes.

Usage:
    # Evaluate the simple BC policy from Step 6
    python 07_evaluate_policy.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # Evaluate with more episodes
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --num_rollouts 50

    # Evaluate on target (held-out) kitchen scenes
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --split target

    # Save evaluation videos
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --video_path /tmp/eval_videos.mp4

For evaluating official Diffusion Policy / pi-0 / GR00T checkpoints,
use the evaluation scripts from those repos instead (see 06_train_policy.py).
"""

import argparse
import os
import sys
import time

# Force osmesa (CPU offscreen renderer) on Linux/WSL2 -- EGL requires
# /dev/dri device access that is unavailable in WSL environments.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

import robocasa  # noqa: F401
from robocasa.utils.env_utils import create_env


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_policy(checkpoint_path, device):
    """Load a trained policy checkpoint."""
    import torch
    import torch.nn as nn

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]

    class SimplePolicy(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            )

        def forward(self, state):
            return self.net(state)

    model = SimplePolicy(state_dim, action_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded policy from: {checkpoint_path}")
    print(f"  Trained for {checkpoint['epoch']} epochs, loss={checkpoint['loss']:.6f}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")

    return model, checkpoint


def extract_state(obs, state_dim, state_keys=None):
    """
    Extract the policy input state vector.

    If `state_keys` is provided (saved in the checkpoint), we concatenate those
    observation keys in order. Otherwise we fall back to the standard LeRobot
    OpenCabinet `observation.state` ordering.
    """
    if state_keys is not None:
        parts = []
        for key in state_keys:
            if key not in obs:
                raise KeyError(f"Missing observation key '{key}' required by the checkpoint")
            parts.append(np.asarray(obs[key]).ravel())
        state = np.concatenate(parts).astype(np.float32)
    else:
        # LeRobot `observation.state` ordering (verified vs dataset parquet):
        #   base_pos (3)
        #   base_quat (4)
        #   base_to_eef_pos (3)
        #   base_to_eef_quat (4)
        #   gripper_qpos (2)
        preferred_keys = [
            "robot0_base_pos",
            "robot0_base_quat",
            "robot0_base_to_eef_pos",
            "robot0_base_to_eef_quat",
            "robot0_gripper_qpos",
        ]

        if all(k in obs for k in preferred_keys):
            state = np.concatenate([obs[k].ravel() for k in preferred_keys]).astype(
                np.float32
            )
        else:
            # Fallback: flatten all non-image ndarray observations.
            state_parts = []
            flat_keys = sorted(
                k
                for k in obs.keys()
                if not k.endswith("_image") and isinstance(obs[k], np.ndarray)
            )
            for key in flat_keys:
                state_parts.append(obs[key].ravel())

            if not state_parts:
                return np.zeros(state_dim, dtype=np.float32)

            state = np.concatenate(state_parts).astype(np.float32)

    # Pad or truncate to match expected state_dim
    if len(state) < state_dim:
        state = np.pad(state, (0, state_dim - len(state)))
    elif len(state) > state_dim:
        state = state[:state_dim]

    return state


def lerobot_action_to_env_action(action):
    """
    Convert a LeRobot-format action vector to the robosuite OpenCabinet env action.

    LeRobot action ordering is defined by the dataset's `meta/modality.json`:
      [ base_motion(4), control_mode(1), end_effector_position(3),
        end_effector_rotation(3), gripper_close(1) ]

    robosuite OpenCabinet (PandaOmron) expects:
      [ arm_pos(3), arm_rot(3), gripper(1), base_motion(3), torso(1), control_mode(1) ]
    """
    a = np.asarray(action, dtype=np.float32).ravel()
    if a.shape[0] < 12:
        a = np.pad(a, (0, 12 - a.shape[0]))
    elif a.shape[0] > 12:
        a = a[:12]

    base_motion = a[0:4]
    control_mode = a[4:5]
    eef_pos = a[5:8]
    eef_rot = a[8:11]
    gripper = a[11:12]

    env_action = np.zeros(12, dtype=np.float32)
    env_action[0:3] = eef_pos
    env_action[3:6] = eef_rot
    env_action[6:7] = gripper
    env_action[7:10] = base_motion[0:3]
    env_action[10:11] = base_motion[3:4]
    env_action[11:12] = control_mode
    return env_action


def run_evaluation(
    model,
    checkpoint,
    num_rollouts,
    max_steps,
    split,
    video_path,
    seed,
):
    """Run evaluation rollouts and collect statistics."""
    import torch
    import imageio

    device = next(model.parameters()).device

    state_dim = int(checkpoint["state_dim"])
    action_dim = int(checkpoint["action_dim"])
    state_keys = checkpoint.get("state_keys", None)
    action_format = checkpoint.get("action_format", "lerobot")

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=256,
        camera_heights=256,
    )

    video_writer = None
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    results = {
        "successes": [],
        "episode_lengths": [],
        "rewards": [],
    }

    for ep in range(num_rollouts):
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")

        ep_reward = 0.0
        success = False

        for step in range(max_steps):
            # Extract state and predict action
            state = extract_state(obs, state_dim, state_keys=state_keys)
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                action = model(state_tensor).cpu().numpy().squeeze(0)

            if action_format == "lerobot":
                # Model outputs LeRobot ordering; convert to env action ordering.
                env_action = lerobot_action_to_env_action(action)
            elif action_format == "env":
                # Model already outputs env action ordering.
                env_action = np.asarray(action, dtype=np.float32).ravel()
                if env_action.shape[0] < 12:
                    env_action = np.pad(env_action, (0, 12 - env_action.shape[0]))
                elif env_action.shape[0] > 12:
                    env_action = env_action[:12]
            else:
                raise ValueError(
                    f"Unknown checkpoint action_format='{action_format}'. Expected 'lerobot' or 'env'."
                )

            # Post-process: RoboCasa demos use arm control only and keep base fixed.
            env_action[11] = -1.0  # control_mode: arm
            env_action[6] = -1.0 if env_action[6] < 0.0 else 1.0  # binary gripper
            env_action[7:11] = 0.0  # base_motion(3) + torso(1)

            obs, reward, done, info = env.step(env_action)
            ep_reward += reward

            if video_writer is not None:
                frame = env.sim.render(
                    height=512, width=768, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(frame)

            if env._check_success():
                success = True
                break

        results["successes"].append(success)
        results["episode_lengths"].append(step + 1)
        results["rewards"].append(ep_reward)

        status = "SUCCESS" if success else "FAIL"
        print(
            f"  Episode {ep + 1:3d}/{num_rollouts}: {status:7s} "
            f"(steps={step + 1:4d}, reward={ep_reward:.1f}) "
            f'layout={env.layout_id}, style={env.style_id}, task="{lang}"'
        )

    if video_writer:
        video_writer.close()

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained OpenCabinet policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint (.pt file)",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Max steps per episode"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
        help="Kitchen scene split to evaluate on",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save evaluation video (optional)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Policy Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the trained policy
    model, checkpoint = load_policy(args.checkpoint, device)

    # Run evaluation
    print_section(f"Evaluating on {args.split} split ({args.num_rollouts} episodes)")

    results = run_evaluation(
        model=model,
        checkpoint=checkpoint,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        video_path=args.video_path,
        seed=args.seed,
    )

    # Print summary
    print_section("Evaluation Results")

    num_success = sum(results["successes"])
    success_rate = num_success / args.num_rollouts * 100
    avg_length = np.mean(results["episode_lengths"])
    avg_reward = np.mean(results["rewards"])

    print(f"  Split:          {args.split}")
    print(f"  Episodes:       {args.num_rollouts}")
    print(f"  Successes:      {num_success}/{args.num_rollouts}")
    print(f"  Success rate:   {success_rate:.1f}%")
    print(f"  Avg ep length:  {avg_length:.1f} steps")
    print(f"  Avg reward:     {avg_reward:.3f}")

    if args.video_path:
        print(f"\n  Video saved to: {args.video_path}")

    # Context for expected performance
    print_section("Performance Context")
    print(
        "Expected success rates from the RoboCasa benchmark:\n"
        "\n"
        "  Method            | Pretrain | Target\n"
        "  ------------------|----------|-------\n"
        "  Random actions    |    ~0%   |   ~0%\n"
        "  Diffusion Policy  |  ~30-60% | ~20-50%\n"
        "  pi-0              |  ~40-70% | ~30-60%\n"
        "  GR00T N1.5        |  ~35-65% | ~25-55%\n"
        "\n"
        "Note: The simple MLP policy from Step 6 is not expected to\n"
        "achieve meaningful success rates. Use the official Diffusion\n"
        "Policy repo for real results."
    )


if __name__ == "__main__":
    main()
