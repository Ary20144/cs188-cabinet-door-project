"""
Step 6: Train a Policy
========================
This script provides a self-contained training loop for a simple
behavior-cloning policy on the OpenCabinet task, suitable for
understanding the training pipeline.

For production-quality training, use the official Diffusion Policy repo:
    git clone https://github.com/robocasa-benchmark/diffusion_policy
    cd diffusion_policy && pip install -e .
    python train.py --config-name=train_diffusion_transformer_bs192 task=robocasa/OpenCabinet

This simplified version trains a small MLP policy to demonstrate
the data loading -> training -> checkpoint pipeline.

Usage:
    python 06_train_policy.py [--epochs 50] [--batch_size 32] [--lr 1e-4]
    python 06_train_policy.py --use_diffusion_policy   # Use official repo
"""

import argparse
import os
import sys
import yaml

import numpy as np


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_path():
    """Get the path to the OpenCabinet dataset."""
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


def build_augmented_state_action_pairs(
    dataset_dir,
    state_keys,
    max_episodes=None,
    max_steps_per_episode=None,
):
    """
    Build (state, action) pairs by replaying expert actions in the simulator and
    extracting additional environment observations (e.g. door-relative features)
    that are not stored in `observation.state`.
    """
    from pathlib import Path
    import json

    import robosuite
    import robocasa.utils.lerobot_utils as LU
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    dataset_dir = Path(dataset_dir)
    episodes = LU.get_episodes(dataset_dir)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    env_meta = LU.get_env_metadata(dataset_dir)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs["renderer"] = "mjviewer"

    env = robosuite.make(**env_kwargs)

    states = []
    actions = []

    for ep_num, _ep_dir in enumerate(episodes):
        ep_states = LU.get_episode_states(dataset_dir, ep_num)
        ep_actions = LU.get_episode_actions(dataset_dir, ep_num)

        if max_steps_per_episode is not None:
            ep_states = ep_states[:max_steps_per_episode]
            ep_actions = ep_actions[:max_steps_per_episode]

        init_state = dict(states=ep_states[0])
        init_state["model"] = LU.get_episode_model_xml(dataset_dir, ep_num)
        init_state["ep_meta"] = json.dumps(LU.get_episode_meta(dataset_dir, ep_num))
        reset_to(env, init_state)

        for t in range(ep_actions.shape[0]):
            obs = env._get_observations(force_update=True)
            parts = []
            for key in state_keys:
                val = obs.get(key)
                if val is None:
                    raise KeyError(f"Missing observation key '{key}' in env obs")
                parts.append(np.asarray(val).ravel())
            states.append(np.concatenate(parts, dtype=np.float32))
            actions.append(ep_actions[t].astype(np.float32))
            env.step(ep_actions[t])

    env.close()
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def train_simple_policy(config):
    """
    Train a simple behavior-cloning policy.

    This is a simplified training loop to illustrate the pipeline.
    For real results, use the official Diffusion Policy codebase.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    print_section("Simple Behavior Cloning Policy")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    # ----------------------------------------------------------------
    # 1. Build dataset
    # ----------------------------------------------------------------
    print("\nLoading dataset...")

    augmented_state = bool(config.get("augmented_state", False))
    max_episodes = config.get("max_episodes", None)
    max_steps_per_episode = config.get("max_steps_per_episode", None)

    if augmented_state:
        print("Using augmented simulator observations (door-relative features).")
        state_keys = [
            "robot0_base_to_eef_pos",
            "robot0_base_to_eef_quat",
            "robot0_gripper_qpos",
            "door_obj_to_robot0_eef_pos",
            "door_obj_to_robot0_eef_quat",
        ]
        states_np, actions_np = build_augmented_state_action_pairs(
            dataset_dir=dataset_path,
            state_keys=state_keys,
            max_episodes=max_episodes,
            max_steps_per_episode=max_steps_per_episode,
        )
        action_format = "env"
    else:
        # Use the compact LeRobot low-dim state stored in parquet: `observation.state` (16 dims)
        # and the raw LeRobot action vector (12 dims) stored in the dataset ordering.
        import pyarrow.parquet as pq

        state_keys = [
            "robot0_base_pos",
            "robot0_base_quat",
            "robot0_base_to_eef_pos",
            "robot0_base_to_eef_quat",
            "robot0_gripper_qpos",
        ]
        action_format = "lerobot"

        # The dataset path from get_ds_path may point to the lerobot dir directly
        # or to the parent. Try both layouts.
        data_dir = os.path.join(dataset_path, "data")
        if not os.path.exists(data_dir):
            data_dir = os.path.join(dataset_path, "lerobot", "data")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Data directory not found under: {dataset_path}\n"
                "Make sure you downloaded the dataset with 04_download_dataset.py"
            )

        chunk_dir = os.path.join(data_dir, "chunk-000")
        if not os.path.exists(chunk_dir):
            raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

        parquet_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".parquet"))
        if max_episodes is not None:
            parquet_files = parquet_files[:max_episodes]
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {chunk_dir}")

        # Fast path: these columns are fixed-size lists, so we can reshape the
        # underlying values buffer directly (much faster than iterating rows).
        states_list = []
        actions_list = []
        for i, pf in enumerate(parquet_files):
            if (i == 0) or ((i + 1) % 10 == 0) or (i + 1 == len(parquet_files)):
                print(f"  Reading {i + 1:3d}/{len(parquet_files)}: {pf}")

            table = pq.read_table(
                os.path.join(chunk_dir, pf),
                columns=["observation.state", "action"],
            )

            state_col = table["observation.state"].combine_chunks()
            action_col = table["action"].combine_chunks()

            state_dim = int(getattr(state_col.type, "list_size", 16))
            action_dim = int(getattr(action_col.type, "list_size", 12))

            s = state_col.values.to_numpy(zero_copy_only=False).reshape(-1, state_dim)
            a = action_col.values.to_numpy(zero_copy_only=False).reshape(-1, action_dim)

            states_list.append(s.astype(np.float32, copy=False))
            actions_list.append(a.astype(np.float32, copy=False))

        if len(states_list) == 0:
            raise RuntimeError("Could not extract (observation.state, action) pairs from parquet files.")

        states_np = np.concatenate(states_list, axis=0)
        actions_np = np.concatenate(actions_list, axis=0)

    print(f"Loaded {len(states_np)} state-action pairs")
    print(f"State dim:  {states_np.shape[-1]}")
    print(f"Action dim: {actions_np.shape[-1]}")

    class ArrayDataset(Dataset):
        def __init__(self, states, actions):
            self.states = states
            self.actions = actions

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return torch.from_numpy(self.states[idx]), torch.from_numpy(self.actions[idx])

    dataset = ArrayDataset(states_np, actions_np)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # ----------------------------------------------------------------
    # 2. Define a simple MLP policy
    # ----------------------------------------------------------------
    state_dim = states_np.shape[-1]
    action_dim = actions_np.shape[-1]

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = SimplePolicy(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # ----------------------------------------------------------------
    # 3. Training loop
    # ----------------------------------------------------------------
    print_section("Training")
    print(f"Epochs:     {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"LR:         {config['learning_rate']}")

    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    avg_loss = float("inf")
    ckpt_path = os.path.join(checkpoint_dir, "best_policy.pt")
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        for states_batch, actions_batch in dataloader:
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)

            pred_actions = model(states_batch)
            loss = nn.functional.mse_loss(pred_actions, actions_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:4d}/{config['epochs']}  Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(checkpoint_dir, "best_policy.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "state_keys": state_keys,
                    "action_format": action_format,
                },
                ckpt_path,
            )

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "final_policy.pt")
    torch.save(
        {
            "epoch": config["epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "state_keys": state_keys,
            "action_format": action_format,
        },
        final_path,
    )

    print(f"\nTraining complete!")
    print(f"Best loss:        {best_loss:.6f}")
    print(f"Best checkpoint:  {ckpt_path}")
    print(f"Final checkpoint: {final_path}")

    print_section("Next Steps")
    print(
        "This simple MLP policy is for educational purposes only.\n"
        "For a policy that can actually solve the task, use the\n"
        "official Diffusion Policy codebase:\n"
        "\n"
        "  git clone https://github.com/robocasa-benchmark/diffusion_policy\n"
        "  cd diffusion_policy && pip install -e .\n"
        "  python train.py \\\n"
        "    --config-name=train_diffusion_transformer_bs192 \\\n"
        "    task=robocasa/OpenCabinet\n"
        "\n"
        "Alternatively, try pi-0 or GR00T N1.5:\n"
        "  https://github.com/robocasa-benchmark/openpi\n"
        "  https://github.com/robocasa-benchmark/Isaac-GR00T"
    )


def print_diffusion_policy_instructions():
    """Print instructions for using the official Diffusion Policy repo."""
    print_section("Official Diffusion Policy Training")
    print(
        "For production-quality policy training, use the official repos:\n"
        "\n"
        "Option A: Diffusion Policy (recommended for single-task)\n"
        "  git clone https://github.com/robocasa-benchmark/diffusion_policy\n"
        "  cd diffusion_policy && pip install -e .\n"
        "\n"
        "  # Train\n"
        "  python train.py \\\n"
        "    --config-name=train_diffusion_transformer_bs192 \\\n"
        "    task=robocasa/OpenCabinet\n"
        "\n"
        "  # Evaluate\n"
        "  python eval_robocasa.py \\\n"
        "    --checkpoint <path-to-checkpoint> \\\n"
        "    --task_set atomic \\\n"
        "    --split target\n"
        "\n"
        "Option B: pi-0 via OpenPi (for foundation model fine-tuning)\n"
        "  git clone https://github.com/robocasa-benchmark/openpi\n"
        "  cd openpi && pip install -e . && pip install -e packages/openpi-client/\n"
        "\n"
        "  XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \\\n"
        "    robocasa_OpenCabinet --exp-name=cabinet_door\n"
        "\n"
        "Option C: GR00T N1.5 (NVIDIA foundation model)\n"
        "  git clone https://github.com/robocasa-benchmark/Isaac-GR00T\n"
        "  cd groot && pip install -e .\n"
        "\n"
        "  python scripts/gr00t_finetune.py \\\n"
        "    --output-dir experiments/cabinet_door \\\n"
        "    --dataset_soup robocasa_OpenCabinet \\\n"
        "    --max_steps 50000\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Train a policy for OpenCabinet")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--augmented_state",
        action="store_true",
        help="Train on richer simulator observations (door-relative features).",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit number of dataset episodes used (default: all).",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=None,
        help="Limit timesteps per episode for augmented-state preprocessing.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/tmp/cabinet_policy_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other args)",
    )
    parser.add_argument(
        "--use_diffusion_policy",
        action="store_true",
        help="Print instructions for using the official Diffusion Policy repo",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Training")
    print("=" * 60)

    if args.use_diffusion_policy:
        print_diffusion_policy_instructions()
        return

    # Build config from args or YAML file
    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "checkpoint_dir": args.checkpoint_dir,
            "augmented_state": args.augmented_state,
            "max_episodes": args.max_episodes,
            "max_steps_per_episode": args.max_steps_per_episode,
        }

    train_simple_policy(config)


if __name__ == "__main__":
    main()
