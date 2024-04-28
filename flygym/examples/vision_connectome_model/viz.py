import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.style
from tqdm import trange
from sys import stderr
from typing import Tuple, List, Dict
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

from flygym.vision import Retina
from flygym.examples.vision_connectome_model import RetinaMapper


matplotlib.style.use("fast")
plt.rcParams["font.family"] = "Arial"


def visualize_vision(
    video_path: Path,
    retina: Retina,
    retina_mapper: RetinaMapper,
    # rendered_image_hist: List[np.ndarray],
    # vision_observation_hist: List[np.ndarray],
    # nn_activities_hist: List[np.ndarray],
    viz_data_all: Tuple[Dict[str, np.ndarray]],
    fps: int,
    figsize: Tuple[float, float] = (12, 9),
    dpi: int = 300,
    cell_activity_range: Tuple[float, float] = (-3, 3),
    cell_activity_cmap: LinearSegmentedColormap = matplotlib.colormaps["seismic"],
) -> FuncAnimation:
    viz_mosaic_pattern = """
    (((...+++++...)))
    (((...+++++...)))
    (((...+++++...)))
    abcdefgh.ABCDEFGH
    ijklmnop.IJKLMNOP
    qrstuvwx.QRSTUVWX
    yz012345.YZ!@#$%^
    67.......&*...///
    """
    cell_panels = {
        "left_cells": "abcdefghijklmnopqrstuvwxyz01234567",
        "right_cells": "ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*",
        "left_input": "(",
        "right_input": ")",
        "birdeye_view": "+",
        "legend": "/",
    }
    cell_order_str = """
    T1    T2    T2a   T3    T4a   T4b   T4c   T4d
    T5a   T5b   T5c   T5d   Tm1   Tm2   Tm3   Tm4
    Tm5Y  Tm5a  Tm5b  Tm5c  Tm9   Tm16  Tm20  Tm28
    Tm30  TmY3  TmY4  TmY5a TmY9  TmY10 TmY13 TmY14
    TmY15 TmY18
    """
    cell_order = cell_order_str.split()

    if type(video_path) is str:
        video_path = Path(video_path)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(
        hspace=0.05, wspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05
    )
    axd = fig.subplot_mosaic(viz_mosaic_pattern)

    # Cached references to plot elements
    plot_elements = {}

    # Function to initialize figure layout
    def init():
        # Turn off all borders
        for ax in axd.values():
            ax.axis("off")

        # Draw legend
        ax_key = cell_panels["legend"]
        ax = axd[ax_key]
        cell_activity_norm = Normalize(*cell_activity_range)
        cell_activity_scalar_mappable = ScalarMappable(
            cmap=cell_activity_cmap, norm=cell_activity_norm
        )
        cell_activity_scalar_mappable.set_array([])
        cbar = plt.colorbar(
            cell_activity_scalar_mappable,
            ax=ax,
            orientation="horizontal",
            shrink=0.8,
            aspect=20,
        )
        cbar.set_ticks(cell_activity_range)
        cbar.set_ticklabels(["hyperpolarization", "depolarization"])

        # Arena birdeye view
        ax_key = cell_panels["birdeye_view"]
        ax = axd[ax_key]
        plot_elements[ax_key] = ax.imshow(
            np.zeros_like(viz_data_all[0]["rendered_image"])
        )

        for side in ("left", "right"):
            # Retina input
            ax_key = cell_panels[f"{side}_input"]
            ax = axd[ax_key]
            plot_elements[ax_key] = ax.imshow(
                np.zeros((retina.nrows, retina.ncols), dtype=np.uint8),
                vmin=0,
                vmax=255,
                cmap="gray",
            )
            ax.set_title(f"Visual Input ({side.title()})")
            # Cell activities
            for ax_key, cell_type in zip(cell_panels[f"{side}_cells"], cell_order):
                ax = axd[ax_key]
                plot_elements[ax_key] = ax.imshow(
                    np.zeros((retina.nrows, retina.ncols)),
                    vmin=cell_activity_range[0],
                    vmax=cell_activity_range[1],
                    cmap=cell_activity_cmap,
                )
                ax.set_title(cell_type)

        return list(plot_elements.values())

    def update(frame_id):
        # Arena birdeye view
        ax_key = cell_panels["birdeye_view"]
        viz_data = viz_data_all[frame_id]
        plot_elements[ax_key].set_data(viz_data["rendered_image"])

        for i_side, side in enumerate(["left", "right"]):
            # Retina input
            visual_input_raw = viz_data["vision_observation"][i_side]
            visual_input_human_readable = retina.hex_pxls_to_human_readable(
                visual_input_raw, color_8bit=True
            ).max(axis=-1)
            ax_key = cell_panels[f"{side}_input"]
            plot_elements[ax_key].set_data(visual_input_human_readable)
            # Cell activities
            for ax_key, cell_type in zip(cell_panels[f"{side}_cells"], cell_order):
                cell_response = viz_data["nn_activities"][cell_type][i_side, :]
                cell_response = retina_mapper.flyvis_to_flygym(cell_response)
                cell_response_human_readable = retina.hex_pxls_to_human_readable(
                    cell_response, color_8bit=False
                )
                plot_elements[ax_key].set_data(cell_response_human_readable)

        return list(plot_elements.values())

    animation = FuncAnimation(
        fig,
        update,
        frames=trange(len(viz_data_all), file=stderr),
        init_func=init,
        blit=False,
    )

    video_path.parent.mkdir(exist_ok=True, parents=True)
    animation.save(video_path, writer="ffmpeg", fps=fps, dpi=dpi)


def save_single_eye_video(
    vision_observation_hist: List[np.ndarray],
    retina: Retina,
    fps: int,
    output_path: Path,
    side: str = "right",
):
    i_side = {"left": 0, "right": 1}[side]
    frames = []
    for vis_obs in vision_observation_hist:
        img = retina.hex_pxls_to_human_readable(
            vis_obs[i_side], color_8bit=True  # right
        ).max(axis=-1)
        frames.append(img)

    # Determine the width and height from the first frame
    height, width = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'mp4v' (or 'avc1') for MP4
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame in frames:
        # Assuming the frames are in BGR color format
        out.write(np.repeat(frame[:, :, None], 3, axis=2))  # Write out frame to video

    out.release()  # Release the VideoWriter


def plot_fly_following_trajectories(
    trajectories_all, radius, leading_fly_speeds, output_path, dt=1e-4
):
    num_steps = np.max([x.shape[0] for x in trajectories_all[("flat", True)]])
    fig, axs = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
    for i, stabilization_on in enumerate([False, True]):
        for j, terrain_type in enumerate(["flat", "blocks"]):
            ax = axs[i, j]
            trajectories = trajectories_all[(terrain_type, stabilization_on)]

            # Draw leading fly
            angular_vel = leading_fly_speeds[terrain_type] / radius
            t_grid = np.arange(num_steps) * dt
            xs = radius * np.sin(angular_vel * t_grid)
            ys = radius * np.cos(angular_vel * t_grid)
            ax.plot(xs, ys, lw=2, color="black", ls="-", label="Leading fly")

            # Draw following flies
            for traj in trajectories:
                ax.plot(traj[:, 0], traj[:, 1], lw=1)

            # Other plot elements
            if stabilization_on:
                stabilization_str = "Head stabilization"
            else:
                stabilization_str = "No head stabilization"
            ax.set_title(f"{terrain_type.title()} terrain\n{stabilization_str}")
            ax.set_aspect("equal")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_xlim(-12.5, 17.5)
            ax.set_ylim(-15, 15)
            if i == 0 and j == 1:
                ax.legend(frameon=False, bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.savefig(output_path)
