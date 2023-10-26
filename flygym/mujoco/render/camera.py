import numpy as np
from typing import Tuple, Union, Optional

from flygym.mujoco import NeuroMechFly


class Camera:
    def __init__(
        self,
        name: str,
        position: Tuple[float, float, float],
        attached_to: Optional[NeuroMechFly],
        fovy: float = 45,
        frame_size: Tuple[int, int] = (640, 480),
        fix_x: bool = False,
        fix_y: bool = False,
        fix_z: bool = False,
        keep_gravity_down: bool = False,
        draw_contacts: bool = False,
        draw_gravity: bool = False,
    ):
        self.name = name
        self.nominal_pos = np.array(position)
        self.attached_fly = attached_to
        self.fovy = fovy
        self.frame_size = frame_size
        self.fix_x = fix_x
        self.fix_y = fix_y
        self.fix_z = fix_z
        self.keep_gravity_down = keep_gravity_down
        self.draw_contacts = draw_contacts
        self.draw_gravity = draw_gravity

        self.curr_pos = np.array(position)

    def render(
        self, physics_handle, contacts=None, gravity_vector=None
    ) -> Union[np.ndarray, None]:
        if self.attached_fly:
            self._update_pos_relative_to_fly(
                physics_handle, self.attached_fly.current_observation["fly"][:3]
            )
        if self.keep_gravity_down:
            self._align_with_gravity(...)
        frame = physics_handle.render(
            width=self.frame_size[0], height=self.frame_size[1], camera_id=self.name
        )
        if self.draw_contacts:
            frame = self._visualize_contacts_forces(frame, contacts)
        if self.draw_gravity:
            frame = self._visualize_gravity_vector(frame, gravity_vector)
        return frame

    def _update_pos_relative_to_fly(
        self, physics_handle, fly_pos: Tuple[float, float, float]
    ) -> None:
        cam_xml = self.attached_fly.find("camera", self.name)
        cam_in_physics = physics_handle.bind(cam_xml)
        tgt_pos = np.array(fly_pos) + self.nominal_pos
        for i, to_fix in enumerate(self.fix_x, self.fix_y, self.fix_z):
            if to_fix:
                tgt_pos[i] = self.nominal_pos[i]
        cam_in_physics.xpos = tgt_pos

    def _align_with_gravity(self, gravity: Tuple[float, float, float]) -> None:
        raise NotImplementedError

    def _visualize_contacts_forces(
        self, frame: np.ndarray, contacts: np.ndarray
    ) -> np.ndarray:
        ...

    def _visualize_gravity_vector(
        self, frame: np.ndarray, gravity_vector: np.ndarray
    ) -> np.ndarray:
        ...
