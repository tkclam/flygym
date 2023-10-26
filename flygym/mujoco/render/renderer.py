import numpy as np
import dm_control.mujoco


class VideoRenderer:
    def __init__(
        self,
        window_size,
        playspeed,
        fps,
        add_timestamp_text,
        add_playspeed_text,
        draw_contact_force,
        decompose_contact_force,
        contact_force_arrow_scaling,
        draw_gravity,
        gravity_arrow_scaling,
        tip_length,
    ):
        self.window_size = window_size
        self.playspeed = playspeed
        self.fps = fps
        self.add_timestamp_text = add_timestamp_text
        self.add_playspeed_text = add_playspeed_text
        self.draw_contact_force = draw_contact_force
        self.decompose_contact_force = decompose_contact_force
        self.contact_force_arrow_scaling = contact_force_arrow_scaling
        self.draw_gravity = draw_gravity
        self.gravity_arrow_scaling = gravity_arrow_scaling
        self.tip_length = tip_length

        self._last_render_time = -np.inf

        self.frames = []

        if draw_contact_force:
            self._last_contact_force = []
            self._last_contact_pos = []
    
        if draw_contact_force or draw_gravity:
            self._dm_camera = dm_control.mujoco.Camera(
                self.physics,
                camera_id=self.sim_params.render_camera,
                width=window_size[0],
                height=window_size[1],
            )
            self._decompose_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

    def need_new_frame(self, curr_time):
        ...

    def add_frame(self, image, visual_input, olfactory_input):
        ...

        self.frames.append(image)