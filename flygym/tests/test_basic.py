import numpy as np
import tempfile
import logging
from pathlib import Path

from flygym import Fly, SingleFlySimulation, Camera
from flygym.util import plot_mujoco_rollout


def test_basic_untethered_sinewave():
    np.random.seed(0)

    sim = SingleFlySimulation(fly=Fly())
    run_time = 0.01
    freq = 20
    amp = np.pi / 2

    obs, _ = sim.reset()
    fly_init_pos = obs["joints"][0]

    rendered_images = []
    obs_list = []
    while sim.curr_time < run_time - 1e-5:
        joint_pos = fly_init_pos + amp * np.sin(freq * sim.curr_time)
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = sim.step(action)
        img = sim.render()
        if img[0] is not None:
            rendered_images.append(img)
        obs_list.append(obs)
    sim.close()

    assert len(obs_list) == int(run_time / sim.timestep)
    assert len(rendered_images) == 2
    assert len(np.unique(rendered_images[-1][0])) > 100  # img has many unique pxl vals

    temp_base_dir = Path(tempfile.gettempdir()) / "flygym_test"
    logging.info(f"temp_base_dir: {temp_base_dir}")
    out_dir = temp_base_dir / "mujoco_basic_untethered_sinewave"
    sim.cameras[0].save_video(out_dir / "video.mp4")
    plot_mujoco_rollout(obs_list, sim.timestep, out_dir / "plot.png")
