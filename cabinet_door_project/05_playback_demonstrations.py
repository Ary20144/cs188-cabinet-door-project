"""
Step 5: Play Back Demonstrations
==================================
Visualizes downloaded demonstrations for the OpenCabinet task.
Watch how an expert opens cabinet doors -- this is the data your
policy will learn from.

Usage:
    # On-screen rendering (Mac: use mjpython)
    mjpython 05_playback_demonstrations.py

    # Off-screen rendering to video
    python 05_playback_demonstrations.py --render_offscreen

    # Play back multiple demos
    python 05_playback_demonstrations.py --num_demos 5 --render_offscreen
"""

import os
import sys

# ── WSLg / XWayland GL setup — re-exec approach ─────────────────────────────
# Same fix as 03_teleop_collect_demos.py: on WSLg the .bashrc often sets a
# stale VcXsrv-style DISPLAY that breaks GLFW.  os.execve() restarts this
# process with correct env vars baked in at the OS level before any C library
# (Mesa, GLFW) is initialized.  Sentinel prevents infinite re-exec.
if sys.platform == "linux" and "__TELEOP_DISPLAY_OK" not in os.environ:
    _env = dict(os.environ)
    _changed = False

    if _env.get("WAYLAND_DISPLAY"):
        if not _env.get("DISPLAY", "").startswith(":"):
            _env["DISPLAY"] = ":0"
            _changed = True
        if _env.get("GALLIUM_DRIVER") != "llvmpipe":
            _env["GALLIUM_DRIVER"] = "llvmpipe"
            _changed = True
        if _env.get("MESA_GL_VERSION_OVERRIDE") != "4.5":
            _env["MESA_GL_VERSION_OVERRIDE"] = "4.5"
            _changed = True

    if _changed:
        _env["__TELEOP_DISPLAY_OK"] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, _env)
    else:
        os.environ["__TELEOP_DISPLAY_OK"] = "1"
# ────────────────────────────────────────────────────────────────────────────

import argparse

import robocasa  # noqa: F401
from robocasa.scripts.download_datasets import download_datasets
from robocasa.scripts.dataset_scripts.playback_dataset import playback_dataset
from robocasa.utils.dataset_registry_utils import get_ds_path
from termcolor import colored


def main():
    parser = argparse.ArgumentParser(description="Play back OpenCabinet demonstrations")
    parser.add_argument(
        "--render_offscreen",
        action="store_true",
        help="Render to video file instead of on-screen window",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/tmp/cabinet_demo_playback",
        help="Directory for off-screen rendered videos",
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=1,
        help="Number of demonstrations to play back",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="human",
        choices=["human", "mg"],
        help="Dataset source: human demonstrations or MimicGen",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Demonstration Playback")
    print("=" * 60)

    task = "OpenCabinet"
    dataset = get_ds_path(task, source=args.source)

    if dataset is None:
        print(f"\nNo registered dataset for task={task}, source={args.source}")
        return

    if not os.path.exists(dataset):
        print(colored(f"\nDataset not found locally. Downloading...", "yellow"))
        download_datasets(tasks=[task], split=["pretrain"], source=[args.source])

    print(f"\nDataset: {dataset}")
    print(f"Source:  {args.source}")
    print(f"Playing back {args.num_demos} demonstration(s)...\n")

    # Configure rendering
    if args.render_offscreen:
        render = False
        os.makedirs(args.video_path, exist_ok=True)
        video_path = os.path.join(args.video_path, "demo_playback.mp4")
        print(f"Saving video to: {video_path}")
    else:
        render = True
        video_path = None

    # Play back demonstrations
    playback_dataset(
        dataset=dataset,
        use_actions=False,
        use_abs_actions=False,
        use_obs=False,
        filter_key=None,
        n=args.num_demos,
        render=render,
        # Use a camera that makes the door motion easy to see.
        render_image_names=["robot0_agentview_right"],
        camera_height=512,
        camera_width=768,
        video_path=video_path,
        video_skip=5,
        extend_states=True,
        first=False,
        verbose=True,
    )

    if args.render_offscreen and video_path:
        print(f"\nVideo saved to: {video_path}")

    print("\nDone! Proceed to 06_train_policy.py")


if __name__ == "__main__":
    main()
