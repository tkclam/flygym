import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import dm_control
import gymnasium as gym
import imageio
import numpy as np
from dm_control import mjcf
from dm_control.utils import transformations
from gymnasium import spaces
from gymnasium.core import ObsType
from scipy.spatial.transform import Rotation as R

import flygym.preprogrammed as preprogrammed
import flygym.state as state
import flygym.util as util
import flygym.vision as vision
from flygym.arena import BaseArena, FlatTerrain
from flygym.core import Parameters
from flygym.util import get_data_path


_roll_eye = np.roll(np.eye(4, 3), -1)


class NeuroMechFlyV0(gym.Env):
    """Deprecated NeuroMechFly environment. Only intended to be used
    for testing purposes.

    Attributes
    ----------
    sim_params : flygym.Parameters
        Parameters of the MuJoCo simulation.
    timestep: float
        Simulation timestep in seconds.
    output_dir : Path
        Directory to save simulation data.
    arena : flygym.arena.BaseArena
        The arena in which the fly is placed.
    spawn_pos : Tuple[float, float, float], optional
        The (x, y, z) position in the arena defining where the fly will be
        spawn, in mm.
    spawn_orientation : Tuple[float, float, float, float], optional
        The spawn orientation of the fly in the Euler angle format: (x, y,
        z), where x, y, z define the rotation around x, y and z in radian.
    control : str
        The joint controller type. Can be "position", "velocity", or
        "torque".
    init_pose : flygym.state.BaseState
        Which initial pose to start the simulation from.
    render_mode : str
        The rendering mode. Can be "saved" or "headless".
    actuated_joints : List[str]
        List of names of actuated joints.
    contact_sensor_placements : List[str]
        List of body segments where contact sensors are placed. By
        default all tarsus segments.
    detect_flip : bool
        If True, the simulation will indicate whether the fly has flipped
        in the ``info`` returned by ``.step(...)``. Flip detection is
        achieved by checking whether the leg tips are free of any contact
        for a duration defined in the configuration file. Flip detection is
        disabled for a period of time at the beginning of the simulation as
        defined in the configuration file. This avoids spurious detection
        when the fly is not standing reliably on the ground yet.
    retina : flygym.vision.Retina
        The retina simulation object used to render the fly's visual
        inputs.
    arena_root = dm_control.mjcf.RootElement
        The root element of the arena.
    physics: dm_control.mjcf.Physics
        The MuJoCo Physics object built from the arena's MJCF model with
        the fly in it.
    curr_time : float
        The (simulated) time elapsed since the last reset (in seconds).
    action_space : gymnasium.core.ObsType
        Definition of the simulation's action space as a Gym environment.
    observation_space : gymnasium.core.ObsType
        Definition of the simulation's observation space as a Gym
        environment.
    model : dm_control.mjcf.RootElement
        The MuJoCo model.
    vision_update_mask : np.ndarray
        The refresh frequency of the visual system is often loser than the
        same as the physics simulation time step. This 1D mask, whose
        size is the same as the number of simulation time steps, indicates
        in which time steps the visual inputs have been refreshed. In other
        words, the visual input frames where this mask is False are
        repetitions of the previous updated visual input frames.
    """

    _config = util.load_config()

    def __init__(
        self,
        sim_params: Parameters = None,
        actuated_joints: List = preprogrammed.all_leg_dofs,
        contact_sensor_placements: List = preprogrammed.all_tarsi_links,
        output_dir: Optional[Path] = None,
        arena: BaseArena = None,
        xml_variant: Union[str, Path] = "seqik",
        spawn_pos: Tuple[float, float, float] = (0.0, 0.0, 0.5),
        spawn_orientation: Tuple[float, float, float] = (0.0, 0.0, np.pi / 2),
        control: str = "position",
        init_pose: Union[str, state.KinematicPose] = "stretch",
        floor_collisions: Union[str, List[str]] = "legs",
        self_collisions: Union[str, List[str]] = "legs",
        detect_flip: bool = False,
    ) -> None:
        """Initialize a NeuroMechFly environment.

        Parameters
        ----------
        sim_params : flygym.Parameters
            Parameters of the MuJoCo simulation.
        actuated_joints : List[str], optional
            List of names of actuated joints. By default all active leg
            DoFs.
        contact_sensor_placements : List[str], optional
            List of body segments where contact sensors are placed. By
            default all tarsus segments.
        output_dir : Path, optional
            Directory to save simulation data. If ``None``, no data will
            be saved. By default None.
        arena : flygym.arena.BaseArena, optional
            The arena in which the fly is placed. ``FlatTerrain`` will be
            used if not specified.
        xml_variant: str or Path, optional
            The variant of the fly model to use. Multiple variants exist
            because when replaying experimentally recorded behavior, the
            ordering of DoF angles in multi-DoF joints depends on how they
            are configured in the upstream inverse kinematics program. Two
            variants are provided: "seqik" (default) and "deepfly3d" (for
            legacy data produced by DeepFly3D, Gunel et al., eLife, 2019).
            The ordering of DoFs can be seen from the XML files under
            ``flygym/data/mjcf/``.
        spawn_pos : Tuple[float, float, float], optional
            The (x, y, z) position in the arena defining where the fly
            will be spawn, in mm. By default (0, 0, 0.5).
        spawn_orientation : Tuple[float, float, float], optional
            The spawn orientation of the fly in the Euler angle format:
            (x, y, z), where x, y, z define the rotation around x, y and
            z in radian. By default (0.0, 0.0, pi/2), which leads to a
            position facing the positive direction of the x-axis.
        control : str, optional
            The joint controller type. Can be "position", "velocity", or
            "torque", by default "position".
        init_pose : BaseState, optional
            Which initial pose to start the simulation from. By default
            "stretch" kinematic pose with all legs fully stretched.
        floor_collisions :str
            Which set of collisions should collide with the floor. Can be
            "all", "legs", "tarsi" or a list of body names. By default
            "legs".
        self_collisions : str
            Which set of collisions should collide with each other. Can be
            "all", "legs", "legs-no-coxa", "tarsi", "none", or a list of
            body names. By default "legs".
        detect_flip : bool
            If True, the simulation will indicate whether the fly has
            flipped in the ``info`` returned by ``.step(...)``. Flip
            detection is achieved by checking whether the leg tips are free
            of any contact for a duration defined in the configuration
            file. Flip detection is disabled for a period of time at the
            beginning of the simulation as defined in the configuration
            file. This avoids spurious detection when the fly is not
            standing reliably on the ground yet. By default False.
        """
        if sim_params is None:
            sim_params = Parameters()
        if arena is None:
            arena = FlatTerrain()
        self.sim_params = deepcopy(sim_params)
        self.actuated_joints = actuated_joints

        if self.sim_params.head_stabilization_kp != 0:
            assert (
                "joint_Head_yaw" not in self.actuated_joints
                and "joint_Head" not in self.actuated_joints
            ), (
                "Head stabilization is not compatible with head joints "
                "being in the actuated joints list."
            )

        self.contact_sensor_placements = contact_sensor_placements
        self.timestep = sim_params.timestep
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.arena = arena
        self.spawn_pos = spawn_pos
        # convert to mujoco orientation format [0, 0, 0] would orient along the x-axis
        # but the output fly_orientation from framequat would be [0, 0, pi/2] for
        # spawn_orient = [0, 0, 0]
        self.spawn_orientation = spawn_orientation - np.array((0, 0, np.pi / 2))
        self.control = control
        if isinstance(init_pose, str):
            init_pose = preprogrammed.get_preprogrammed_pose(init_pose)
        self.init_pose = init_pose
        self.render_mode = sim_params.render_mode
        self.detect_flip = detect_flip
        self._last_tarsalseg_names = [
            f"{side}{pos}Tarsus5" for side in "LR" for pos in "FMH"
        ]

        if self.sim_params.draw_contacts and "cv2" not in sys.modules:
            logging.warning(
                "Overriding `draw_contacts` to False because OpenCV is required "
                "to draw the arrows but it is not installed."
            )
            self.sim_params.draw_contacts = False

        # Parse collisions specs
        if isinstance(floor_collisions, str):
            self._floor_collisions = preprogrammed.get_collision_geometries(
                floor_collisions
            )
        else:
            self._floor_collisions = floor_collisions
        if isinstance(self_collisions, str):
            self._self_collisions = preprogrammed.get_collision_geometries(
                self_collisions
            )
        else:
            self._self_collisions = self_collisions

        self.n_legs = 6

        self._last_adhesion = np.zeros(self.n_legs)
        self._active_adhesion = np.zeros(self.n_legs)

        if self.sim_params.draw_adhesion and not self.sim_params.enable_adhesion:
            logging.warning(
                "Overriding `draw_adhesion` to False because adhesion is not enabled."
            )
            self.sim_params.draw_adhesion = False

        if self.sim_params.draw_adhesion:
            self._leg_adhesion_drawing_segments = np.array(
                [
                    ["Animat/" + tarsus5.replace("5", str(i)) for i in range(1, 6)]
                    for tarsus5 in self._last_tarsalseg_names
                ]
            ).astype("U64")
            self._adhesion_rgba = [1.0, 0.0, 0.0, 0.8]
            self._active_adhesion_rgba = [0.0, 0.0, 1.0, 0.8]
            self._base_rgba = [0.5, 0.5, 0.5, 1.0]

        if self.sim_params.draw_gravity:
            self._last_fly_pos = spawn_pos
            self._gravity_rgba = [1 - 213 / 255, 1 - 90 / 255, 1 - 255 / 255, 1.0]
            cam_name = self.sim_params.render_camera
            self._arrow_offset = np.zeros(3)
            if "bottom" in cam_name or "top" in cam_name:
                self._arrow_offset[0] = -3
                self._arrow_offset[1] = 2
            elif "left" in cam_name or "right" in cam_name:
                self._arrow_offset[2] = 2
                self._arrow_offset[0] = -3
            elif "front" in cam_name or "back" in cam_name:
                self._arrow_offset[2] = 2
                self._arrow_offset[1] = 3

        if self.sim_params.align_camera_with_gravity:
            self._camera_rot = np.eye(3)

        # Load NMF model
        if isinstance(xml_variant, str):
            xml_variant = (
                get_data_path("flygym", "data")
                / self._config["paths"]["mjcf"][xml_variant]
            )
        self.model = mjcf.from_path(xml_variant)
        self._set_geom_colors()

        # Add cameras imitating the fly's eyes
        self._curr_visual_input = None
        self._curr_raw_visual_input = None
        self._last_vision_update_time = -np.inf
        self._eff_visual_render_interval = 1 / self.sim_params.vision_refresh_rate
        self._vision_update_mask: List[bool] = []
        if self.sim_params.enable_vision:
            self._configure_eyes()
            self.retina = vision.Retina()

        # Define list of actuated joints
        self._actuators = [
            self.model.find("actuator", f"actuator_{control}_{joint}")
            for joint in actuated_joints
        ]

        self._head_stabilization_actuators = [
            self.model.find("actuator", "actuator_position_joint_Head_yaw"),  # roll
            self.model.find("actuator", "actuator_position_joint_Head"),  # pitch
        ]

        self._set_actuators_gain()
        self._set_geoms_friction()
        self._set_joints_stiffness_and_damping()
        self._set_compliant_tarsus()

        self._thorax = self.model.find("body", "Thorax")

        self._floor_height = self._get_max_floor_height(arena)

        # Add arena and put fly in it
        arena.spawn_entity(self.model, self.spawn_pos, self.spawn_orientation)
        self.arena_root = arena.root_element
        self.arena_root.option.timestep = self.timestep

        camera_name = self.sim_params.render_camera
        model_camera_name = self.sim_params.render_camera.split("/")[-1]
        self._cam = self.model.find("camera", model_camera_name)
        self._initialize_custom_camera_handling(camera_name)

        # Add collision/contacts
        floor_collision_geoms = self._parse_collision_specs(floor_collisions)
        self._floor_contacts, self._floor_contact_names = self._define_floor_contacts(
            floor_collision_geoms
        )
        self_collision_geoms = self._parse_collision_specs(self_collisions)
        self._self_contacts, self._self_contact_names = self._define_self_contacts(
            self_collision_geoms
        )

        # Add sensors
        self._joint_sensors = self._add_joint_sensors()
        self._body_sensors = self._add_body_sensors()
        self._end_effector_sensors = self._add_end_effector_sensors()
        self._antennae_sensors = (
            self._add_odor_sensors() if sim_params.enable_olfaction else None
        )
        self._add_force_sensors()
        self.contact_sensor_placements = [
            f"Animat/{body}" for body in self.contact_sensor_placements
        ]
        self._adhesion_actuators = self._add_adhesion_actuators(
            self.sim_params.adhesion_force
        )
        # Those need to be in the same order as the adhesion sensor
        # (due to comparison with the last adhesion_signal)
        adhesion_sensor_indices = []
        for adhesion_actuator in self._adhesion_actuators:
            for index, contact_sensor in enumerate(self.contact_sensor_placements):
                if f"{contact_sensor}_adhesion" in f"Animat/{adhesion_actuator.name}":
                    adhesion_sensor_indices.append(index)
        self._adhesion_bodies_with_contact_sensors = np.array(adhesion_sensor_indices)

        # Set up physics and apply ad hoc changes to gravity, stiffness, and friction
        self.physics = mjcf.Physics.from_mjcf_model(self.arena_root)
        self._adhesion_actuator_geomid = np.array(
            [
                self.physics.model.geom("Animat/" + adhesion_actuator.body).id
                for adhesion_actuator in self._adhesion_actuators
            ]
        )

        # Set gravity
        self._set_gravity(self.sim_params.gravity)

        # Apply initial pose.(TARSI MUST HAVE MADE COMPLIANT BEFORE)!
        self._set_init_pose(self.init_pose)

        # Set up a few things for rendering
        self.curr_time = 0
        self._last_render_time = -np.inf
        if sim_params.render_mode != "headless":
            self._eff_render_interval = (
                sim_params.render_playspeed / self.sim_params.render_fps
            )
        self._frames = []

        if self.sim_params.draw_contacts:
            self._last_contact_force = []
            self._last_contact_pos = []

        if self.sim_params.draw_contacts or self.sim_params.draw_gravity:
            width, height = self.sim_params.render_window_size
            self._dm_camera = dm_control.mujoco.Camera(
                self.physics,
                camera_id=self.sim_params.render_camera,
                width=width,
                height=height,
            )
            self._decompose_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

        # flip detection
        self._flip_counter = 0

        # Define action and observation spaces
        action_bound = np.pi if self.control == "position" else np.inf
        self.action_space = self._define_action_space(action_bound)
        self.observation_space = self._define_observation_space()

        # Add metadata as specified by Gym
        self.metadata = {
            "render_modes": ["saved", "headless"],
            "render_fps": sim_params.render_fps,
        }

    def _configure_eyes(self):
        for name in ["LEye_cam", "REye_cam"]:
            sensor_config = self._config["vision"]["sensor_positions"][name]
            parent_body = self.model.find("body", sensor_config["parent"])
            sensor_body = parent_body.add(
                "body", name=f"{name}_body", pos=sensor_config["rel_pos"]
            )
            sensor_body.add(
                "camera",
                name=name,
                dclass="nmf",
                mode="fixed",
                euler=sensor_config["orientation"],
                fovy=self._config["vision"]["fovy_per_eye"],
            )
            if self.sim_params.draw_sensor_markers:
                sensor_body.add(
                    "geom",
                    name=f"{name}_marker",
                    type="sphere",
                    size=[0.06],
                    rgba=sensor_config["marker_rgba"],
                )

        self._geoms_to_hide = self._config["vision"]["hidden_segments"]

    def _parse_collision_specs(self, collision_spec: Union[str, List[str]]):
        if collision_spec == "all":
            return [geom.name for geom in self.model.find_all("geom")]
        elif isinstance(collision_spec, str):
            return preprogrammed.get_collision_geometries(collision_spec)
        elif isinstance(collision_spec, list):
            return collision_spec
        else:
            raise TypeError(
                "Collision specs must be a string ('legs', 'legs-no-coxa', 'tarsi', "
                "'none'), or a list of body segment names."
            )

    def _correct_camera_orientation(self, camera_name: str):
        # Correct the camera orientation by incorporating the spawn rotation
        # of the arena

        # Get the camera
        camera = self.model.find("camera", camera_name)
        if camera is None or camera.mode in ["targetbody", "targetbodycom"]:
            return 0
        if "head" in camera_name or "front_zoomin" in camera_name:
            # Don't correct the head camera
            return camera

        # Add the spawn rotation (keep horizon flat)
        spawn_quat = np.array(
            [
                np.cos(self.spawn_orientation[-1] / 2),
                self.spawn_orientation[0] * np.sin(self.spawn_orientation[-1] / 2),
                self.spawn_orientation[1] * np.sin(self.spawn_orientation[-1] / 2),
                self.spawn_orientation[2] * np.sin(self.spawn_orientation[-1] / 2),
            ]
        )

        # Change camera euler to quaternion
        camera_quat = transformations.euler_to_quat(camera.euler)
        new_camera_quat = transformations.quat_mul(
            transformations.quat_inv(spawn_quat), camera_quat
        )
        camera.euler = transformations.quat_to_euler(new_camera_quat)

        # Elevate the camera slightly gives a better view of the arena
        if not "zoomin" in camera_name:
            camera.pos = camera.pos + [0.0, 0.0, 0.5]
        if "front" in camera_name:
            camera.pos[2] = camera.pos[2] + 1.0

        return camera

    def _set_geom_colors(self):
        for type_, specs in self._config["appearance"].items():
            # Define texture and material
            if specs["texture"] is not None:
                self.model.asset.add(
                    "texture",
                    name=f"{type_}_texture",
                    builtin=specs["texture"]["builtin"],
                    mark="random",
                    width=specs["texture"]["size"],
                    height=specs["texture"]["size"],
                    random=specs["texture"]["random"],
                    rgb1=specs["texture"]["rgb1"],
                    rgb2=specs["texture"]["rgb2"],
                    markrgb=specs["texture"]["markrgb"],
                )
            self.model.asset.add(
                "material",
                name=f"{type_}_material",
                texture=f"{type_}_texture" if specs["texture"] is not None else None,
                rgba=specs["material"]["rgba"],
                specular=0.0,
                shininess=0.0,
                reflectance=0.0,
                texuniform=True,
            )
            # Apply to geoms
            for segment in specs["apply_to"]:
                geom = self.model.find("geom", segment)
                if geom is None:
                    geom = self.model.find("geom", f"{segment}")
                geom.material = f"{type_}_material"

    def _get_max_floor_height(self, arena):
        max_floor_height = -1 * np.inf
        for geom in arena.root_element.find_all("geom"):
            name = geom.name
            if name is None or (
                "floor" in name or "ground" in name or "treadmill" in name
            ):
                if geom.type == "box":
                    block_height = geom.pos[2] + geom.size[2]
                    max_floor_height = max(max_floor_height, block_height)
                elif geom.type == "plane":
                    try:
                        plane_height = geom.pos[2]
                    except TypeError:
                        plane_height = 0.0
                    max_floor_height = max(max_floor_height, plane_height)
                elif geom.type == "sphere":
                    sphere_height = geom.parent.pos[2] + geom.size[0]
                    max_floor_height = max(max_floor_height, sphere_height)
        if np.isinf(max_floor_height):
            max_floor_height = self.spawn_pos[2]
        return max_floor_height

    def _define_action_space(self, action_bound):
        _action_space = {
            "joints": spaces.Box(
                low=-action_bound, high=action_bound, shape=(len(self.actuated_joints),)
            )
        }
        if self.sim_params.enable_adhesion:
            # 0: no adhesion, 1: adhesion
            _action_space["adhesion"] = spaces.Discrete(n=2, start=0)
        return spaces.Dict(_action_space)

    def _define_observation_space(self):
        _observation_space = {
            "joints": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3, len(self.actuated_joints))
            ),
            "fly": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3)),
            "contact_forces": spaces.Box(
                low=-np.inf, high=np.inf, shape=(len(self.contact_sensor_placements), 3)
            ),
            # x, y, z positions of the end effectors (tarsus-5 segments)
            "end_effectors": spaces.Box(low=-np.inf, high=np.inf, shape=(6, 3)),
            "fly_orientation": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
        }
        if self.sim_params.enable_vision:
            _observation_space["vision"] = spaces.Box(
                low=0,
                high=255,
                shape=(2, self._config["vision"]["num_ommatidia_per_eye"], 2),
            )
        if self.sim_params.enable_olfaction:
            _observation_space["odor_intensity"] = spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.arena.odor_dimensions, len(self._antennae_sensors)),
            )
        return spaces.Dict(_observation_space)

    def _initialize_custom_camera_handling(self, camera_name):
        """
        This function is called when the camera is initialized. It can be
        used to customize the camera behavior. I case update_camera_pos is
        True and the camera is within the animat and not a head camera, the
        z position will be fixed to avoid oscillations. If
        self.sim_params.camera_follows_fly_orientation is True, the camera
        will be rotated to follow the fly orientation (i.e. the front
        camera will always be in front of the fly).
        """

        is_Animat = "Animat" in camera_name
        is_visualization_camera = (
            "head" in camera_name
            or "Tarsus" in camera_name
            or "camera_front_zoomin" in camera_name
        )

        is_compound_camera = not camera_name in [
            "Animat/camera_front",
            "Animat/camera_top",
            "Animat/camera_bottom",
            "Animat/camera_back",
            "Animat/camera_right",
            "Animat/camera_left",
        ]

        # always add pos update if it is a head camera
        if is_Animat and not is_visualization_camera:
            self.update_camera_pos = True
            self.cam_offset = self._cam.pos
            if is_compound_camera and self.sim_params.camera_follows_fly_orientation:
                self.sim_params.camera_follows_fly_orientation = False
                logging.warning(
                    "Overriding `camera_follows_fly_orientation` to False because"
                    "it is never applied to visualization cameras (head, tarsus, ect)"
                    "or non Animat camera."
                )
            elif self.sim_params.camera_follows_fly_orientation:
                # Why would that be xyz and not XYZ ? DOES NOT MAKE SENSE BUT IT WORKS
                self.base_camera_rot = R.from_euler(
                    "xyz", self._cam.euler + self.spawn_orientation
                ).as_matrix()
                # THIS SOMEHOW REPLICATES THE CAMERA XMAT OBTAINED BY MUJOCO WHE USING
                # TRACKED CAMERA
            else:
                # if not camera_follows_fly_orientation need to change the camera mode
                # to track
                self._cam.mode = "track"
            return
        else:
            self.update_camera_pos = False
            if self.sim_params.camera_follows_fly_orientation:
                self.sim_params.camera_follows_fly_orientation = False
                logging.warning(
                    "Overriding `camera_follows_fly_orientation` to False because"
                    "it is never applied to visualization cameras (head, tarsus, ect)"
                    "or non Animat camera."
                )
            return

    def _set_actuators_gain(self):
        for actuator in self._actuators:
            actuator.kp = self.sim_params.actuator_kp

        if self.sim_params.head_stabilization_kp != 0:
            for actuator in self._head_stabilization_actuators:
                actuator.kp = self.sim_params.head_stabilization_kp

    def _set_geoms_friction(self):
        for geom in self.model.find_all("geom"):
            geom.friction = self.sim_params.friction

    def _set_joints_stiffness_and_damping(self):
        for joint in self.model.find_all("joint"):
            if joint.name in self.actuated_joints:
                joint.stiffness = self.sim_params.joint_stiffness
                joint.damping = self.sim_params.joint_damping
            else:
                joint.stiffness = self.sim_params.non_actuated_joint_stiffness
                joint.damping = self.sim_params.non_actuated_joint_damping

    def _get_real_parent(self, child):
        real_parent = None
        child_name = child.name.split("_")[0]
        parent = child.parent
        if child_name in parent.name:
            real_parent = self._get_real_parent(parent)

        else:
            real_parent = parent.name.split("_")[0]

        assert (
            real_parent is not None
        ), f"Real parent not found for {child_name} but this cannot be"
        return real_parent

    def get_real_children(self, parent):
        real_children = []
        parent_name = parent.name.split("_")[0]
        for child in parent.get_children("body"):
            if parent_name in child.name:
                real_children.extend(self.get_real_children(child.name))

            else:
                real_children.extend([child.name.split("_")[0]])

        return real_children

    def _define_self_contacts(self, self_collisions_geoms):
        self_contact_pairs = []
        self_contact_pairs_names = []
        for geom1 in self_collisions_geoms:
            for geom2 in self_collisions_geoms:
                is_duplicate = f"{geom1}_{geom2}" in self_contact_pairs_names
                if geom1 != geom2 and not is_duplicate:
                    # Do not add contact if the parent bodies have a child parent
                    # relationship
                    body1 = self.model.find("geom", geom1).parent
                    body2 = self.model.find("geom", geom2).parent
                    simple_body1_name = body1.name.split("_")[0]
                    simple_body2_name = body2.name.split("_")[0]

                    body1_children = self.get_real_children(body1)
                    body2_children = self.get_real_children(body2)

                    body1_parent = self._get_real_parent(body1)
                    body2_parent = self._get_real_parent(body2)

                    if not (
                        body1.name == body2.name
                        or simple_body1_name in body2_children
                        or simple_body2_name in body1_children
                        or simple_body1_name == body2_parent
                        or simple_body2_name == body1_parent
                    ):
                        contact_pair = self.model.contact.add(
                            "pair",
                            name=f"{geom1}_{geom2}",
                            geom1=geom1,
                            geom2=geom2,
                            solref=self.sim_params.contact_solref,
                            solimp=self.sim_params.contact_solimp,
                            margin=0.0,  # change margin to avoid penetration
                        )
                        self_contact_pairs.append(contact_pair)
                        self_contact_pairs_names.append(f"{geom1}_{geom2}")
        return self_contact_pairs, self_contact_pairs_names

    def _define_floor_contacts(self, floor_collisions_geoms):
        floor_contact_pairs = []
        floor_contact_pairs_names = []
        ground_id = 0

        for geom in self.arena_root.find_all("geom"):
            if geom.name is None:
                is_ground = True
            elif not geom.dclass is None and geom.dclass.dclass == "nmf":
                is_ground = False
            elif "cam" in geom.name or "sensor" in geom.name:
                is_ground = False
            else:
                is_ground = True
            if is_ground:
                for animat_geom_name in floor_collisions_geoms:
                    if geom.name is None:
                        geom.name = f"groundblock_{ground_id}"
                        ground_id += 1
                    mean_friction = np.mean(
                        [
                            self.sim_params.friction,  # fly friction
                            self.arena.friction,  # arena ground friction
                        ],
                        axis=0,
                    )
                    floor_contact_pair = self.arena_root.contact.add(
                        "pair",
                        name=f"{geom.name}_{animat_geom_name}",
                        geom1=f"Animat/{animat_geom_name}",
                        geom2=f"{geom.name}",
                        solref=self.sim_params.contact_solref,
                        solimp=self.sim_params.contact_solimp,
                        margin=0.0,  # change margin to avoid penetration
                        friction=np.repeat(
                            mean_friction,
                            (2, 1, 2),
                        ),
                    )
                    floor_contact_pairs.append(floor_contact_pair)
                    floor_contact_pairs_names.append(f"{geom.name}_{animat_geom_name}")

        return floor_contact_pairs, floor_contact_pairs_names

    def _add_joint_sensors(self):
        joint_sensors = []
        for joint in self.actuated_joints:
            joint_sensors.extend(
                [
                    self.model.sensor.add(
                        "jointpos", name=f"jointpos_{joint}", joint=joint
                    ),
                    self.model.sensor.add(
                        "jointvel", name=f"jointvel_{joint}", joint=joint
                    ),
                    self.model.sensor.add(
                        "actuatorfrc",
                        name=f"actuatorfrc_position_{joint}",
                        actuator=f"actuator_position_{joint}",
                    ),
                    self.model.sensor.add(
                        "actuatorfrc",
                        name=f"actuatorfrc_velocity_{joint}",
                        actuator=f"actuator_velocity_{joint}",
                    ),
                    self.model.sensor.add(
                        "actuatorfrc",
                        name=f"actuatorfrc_motor_{joint}",
                        actuator=f"actuator_torque_{joint}",
                    ),
                ]
            )
        return joint_sensors

    def _add_body_sensors(self):
        lin_pos_sensor = self.model.sensor.add(
            "framepos", name="thorax_pos", objtype="body", objname="Thorax"
        )
        lin_vel_sensor = self.model.sensor.add(
            "framelinvel", name="thorax_linvel", objtype="body", objname="Thorax"
        )
        ang_pos_sensor = self.model.sensor.add(
            "framequat", name="thorax_quat", objtype="body", objname="Thorax"
        )
        ang_vel_sensor = self.model.sensor.add(
            "frameangvel", name="thorax_angvel", objtype="body", objname="Thorax"
        )
        orient_sensor = self.model.sensor.add(
            "framezaxis", name="thorax_orient", objtype="body", objname="Thorax"
        )
        return [
            lin_pos_sensor,
            lin_vel_sensor,
            ang_pos_sensor,
            ang_vel_sensor,
            orient_sensor,
        ]

    def _add_end_effector_sensors(self):
        end_effector_sensors = []
        for name in self._last_tarsalseg_names:
            sensor = self.model.sensor.add(
                "framepos",
                name=f"{name}_pos",
                objtype="body",
                objname=name,
            )
            end_effector_sensors.append(sensor)
        return end_effector_sensors

    def _add_odor_sensors(self):
        antennae_sensors = []
        for name, specs in self._config["olfaction"]["sensor_positions"].items():
            parent_body = self.model.find("body", specs["parent"])
            sensor_body = parent_body.add(
                "body", name=f"{name}_body", pos=specs["rel_pos"]
            )
            sensor = self.model.sensor.add(
                "framepos",
                name=f"{name}_pos_sensor",
                objtype="body",
                objname=f"{name}_body",
            )
            antennae_sensors.append(sensor)
            if self.sim_params.draw_sensor_markers:
                sensor_body.add(
                    "geom",
                    name=f"{name}_marker",
                    type="sphere",
                    size=[0.06],
                    rgba=specs["marker_rgba"],
                )
        return antennae_sensors

    def _add_force_sensors(self):
        """
        Add force sensors to the tracked bodies
        Without them the cfrc_ext is zero
        Returns
        -------
        All force sensors
        """
        force_sensors = []
        for tracked_geom in self.contact_sensor_placements:
            body = self.model.find("body", tracked_geom)
            site = body.add(
                "site",
                name=f"{tracked_geom}_site",
                pos=[0, 0, 0],
                size=np.ones(3) * 0.005,
            )
            force_sensor = self.model.sensor.add(
                "force", name=f"force_{body.name}", site=site.name
            )
            force_sensors.append(force_sensor)

        return force_sensors

    def _add_adhesion_actuators(self, gain):
        adhesion_actuators = []
        for name in self._last_tarsalseg_names:
            adhesion_actuators.append(
                self.model.actuator.add(
                    "adhesion",
                    name=f"{name}_adhesion",
                    gain=f"{gain}",
                    body=name,
                    ctrlrange="0 1000000",
                    forcerange="-inf inf",
                )
            )
        return adhesion_actuators

    def _set_init_pose(self, init_pose: Dict[str, float]):
        with self.physics.reset_context():
            for i in range(len(self.actuated_joints)):
                curr_joint = self._actuators[i].joint.name
                if (curr_joint in self.actuated_joints) and (curr_joint in init_pose):
                    animat_name = f"Animat/{curr_joint}"
                    self.physics.named.data.qpos[animat_name] = init_pose[curr_joint]

    def _set_compliant_tarsus(self):
        """Set the Tarsus2/3/4/5 to be compliant by setting the stiffness
        and damping to a low value"""
        stiffness = self.sim_params.tarsus_stiffness
        damping = self.sim_params.tarsus_damping
        for side in "LR":
            for pos in "FMH":
                for tarsus_link in range(2, 5 + 1):
                    joint = self.model.find(
                        "joint", f"joint_{side}{pos}Tarsus{tarsus_link}"
                    )
                    joint.stiffness = stiffness
                    joint.damping = damping

    def _set_gravity(self, gravity: List[float], rot_mat: np.ndarray = None) -> None:
        """Set the gravity of the environment. Changing the gravity vector
        might be useful during climbing simulations. The change in the
        camera point of view has been extensively tested for the simple
        cameras (left right top bottom front back) but not for the composed
        ones.

        Parameters
        ----------
        gravity : List[float]
            The gravity vector.
        rot_mat : np.ndarray, optional
            The rotation matrix to align the camera with the gravity vector
             by default None.
        """
        # Only change the angle of the camera if the new gravity vector and the camera
        # angle are compatible
        camera_is_compatible = False
        if (
            "left" in self.sim_params.render_camera
            or "right" in self.sim_params.render_camera
        ):
            if not gravity[1] > 0:
                camera_is_compatible = True
        # elif "top" in self.sim_params.camera_name or "bottom" in
        # self.sim_params.camera_name:
        elif (
            "front" in self.sim_params.render_camera
            or "back" in self.sim_params.render_camera
        ):
            if not gravity[1] > 0:
                camera_is_compatible = True

        if rot_mat is not None and self.sim_params.align_camera_with_gravity:
            self._camera_rot = rot_mat
        elif camera_is_compatible:
            normalised_gravity = (np.array(gravity) / np.linalg.norm(gravity)).reshape(
                (1, 3)
            )
            downward_ref = np.array([0.0, 0.0, -1.0]).reshape((1, 3))

            if (
                not (normalised_gravity == downward_ref).all()
                and self.sim_params.align_camera_with_gravity
            ):
                # Generate a bunch of vectors to help the optimisation algorithm

                random_vectors = np.tile(np.random.rand(10_000), (3, 1)).T
                downward_refs = random_vectors + downward_ref
                gravity_vectors = random_vectors + normalised_gravity
                downward_refs = downward_refs
                gravity_vectors = gravity_vectors
                rot_mult = R.align_vectors(downward_refs, gravity_vectors)[0]

                rot_simple = R.align_vectors(
                    np.reshape(normalised_gravity, (1, 3)),
                    downward_ref.reshape((1, 3)),
                )[0]

                diff_mult = np.linalg.norm(
                    np.dot(rot_mult.as_matrix(), normalised_gravity.T) - downward_ref.T
                )
                diff_simple = np.linalg.norm(
                    np.dot(rot_simple.as_matrix(), normalised_gravity.T)
                    - downward_ref.T
                )
                if diff_mult < diff_simple:
                    rot = rot_mult
                else:
                    rot = rot_simple

                logging.info(
                    f"{normalised_gravity}, "
                    f"{rot.as_euler('xyz')}, "
                    f"{np.dot(rot.as_matrix(), normalised_gravity.T).T}, ",
                    f"{downward_ref}",
                )

                # check if rotation has effect if not remove it
                euler_rot = rot.as_euler("xyz")
                new_euler_rot = np.zeros(3)
                last_rotated_vector = normalised_gravity
                for i in range(0, 3):
                    new_euler_rot[: i + 1] = euler_rot[: i + 1].copy()

                    rotated_vector = (
                        R.from_euler("xyz", new_euler_rot).as_matrix()
                        @ normalised_gravity.T
                    ).T
                    logging.info(
                        f"{euler_rot}, "
                        f"{new_euler_rot}, "
                        f"{rotated_vector}, "
                        f"{last_rotated_vector}"
                    )
                    if np.linalg.norm(rotated_vector - last_rotated_vector) < 1e-2:
                        logging.info("Removing component {i}")
                        euler_rot[i] = 0
                    last_rotated_vector = rotated_vector

                logging.info(str(euler_rot))
                rot = R.from_euler("xyz", euler_rot)
                rot_mat = rot.as_matrix()

                self._camera_rot = rot_mat.T

        self.physics.model.opt.gravity[:] = gravity

    def set_slope(self, slope: float, rot_axis="y"):
        """Set the slope of the environment and modify the camera
        orientation so that gravity is always pointing down. Changing the
        gravity vector might be useful during climbing simulations. The
        change in the camera angle has been extensively tested for the
        simple cameras (left, right, top, bottom, front, back) but not for
        the composed ones.

        Parameters
        ----------
        slope : float
            The desired_slope of the environment in degrees.
        rot_axis : str, optional
            The axis about which the slope is applied, by default "y".
        """
        rot_mat = np.eye(3)
        if rot_axis == "x":
            rot_mat = transformations.rotation_x_axis(np.deg2rad(slope))
        elif rot_axis == "y":
            rot_mat = transformations.rotation_y_axis(np.deg2rad(slope))
        elif rot_axis == "z":
            rot_mat = transformations.rotation_z_axis(np.deg2rad(slope))
        new_gravity = np.dot(rot_mat, self.sim_params.gravity)
        self._set_gravity(new_gravity, rot_mat)

        return 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset the Gym environment.

        Parameters
        ----------
        seed : int
            Random seed for the environment. The provided base simulation
            is deterministic, so this does not have an effect unless
            extended by the user.
        options : Dict
            Additional parameter for the simulation. There is none in the
            provided base simulation, so this does not have an effect
            unless extended by the user.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        Dict[str, Any]
            Any additional information that is not part of the observation.
            This is an empty dictionary by default but the user can
            override this method to return additional information.
        """
        super().reset(seed=seed)
        self.physics.reset()
        if np.any(self.physics.model.opt.gravity[:] - self.sim_params.gravity > 1e-3):
            self._set_gravity(self.sim_params.gravity)
            if self.sim_params.align_camera_with_gravity:
                self._camera_rot = np.eye(3)
        self.curr_time = 0
        self._set_init_pose(self.init_pose)
        self._frames = []
        self._last_render_time = -np.inf
        self._last_vision_update_time = -np.inf
        self._curr_raw_visual_input = None
        self._curr_visual_input = None
        self._vision_update_mask = []
        self._flip_counter = 0
        obs = self.get_observation()
        info = self.get_info()
        if self.sim_params.enable_vision:
            info["vision_updated"] = True
        return obs, info

    def step(
        self, action: ObsType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Step the Gym environment.

        Parameters
        ----------
        action : ObsType
            Action dictionary as defined by the environment's action space.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        float
            The reward as defined by the environment.
        bool
            Whether the episode has terminated due to factors that are
            defined within the Markov Decision Process (e.g. task
            completion/failure, etc.).
        bool
            Whether the episode has terminated due to factors beyond the
            Markov Decision Process (e.g. time limit, etc.).
        Dict[str, Any]
            Any additional information that is not part of the observation.
            This is an empty dictionary by default (except when vision is
            enabled; in this case a "vision_updated" boolean variable
            indicates whether the visual input to the fly was refreshed at
            this step) but the user can override this method to return
            additional information.
        """
        self.arena.step(dt=self.timestep, physics=self.physics)
        self.physics.bind(self._actuators).ctrl = action["joints"]

        if self.sim_params.head_stabilization_kp != 0:
            self._stabilize_head()

        if self.sim_params.enable_adhesion:
            self.physics.bind(self._adhesion_actuators).ctrl = action["adhesion"]
            self._last_adhesion = action["adhesion"]

        self.physics.step()
        self.curr_time += self.timestep
        observation = self.get_observation()
        reward = self.get_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.get_info()
        if self.sim_params.enable_vision:
            vision_updated_this_step = self.curr_time == self._last_vision_update_time
            self._vision_update_mask.append(vision_updated_this_step)
            info["vision_updated"] = vision_updated_this_step

        if self.detect_flip:
            if observation["contact_forces"].sum() < 1:
                self._flip_counter += 1
            else:
                self._flip_counter = 0
            flip_config = self._config["flip_detection"]
            has_passed_init = self.curr_time > flip_config["ignore_period"]
            contact_lost_time = self._flip_counter * self.timestep
            lost_contact_long_enough = contact_lost_time > flip_config["flip_threshold"]
            info["flip"] = has_passed_init and lost_contact_long_enough
            info["flip_counter"] = self._flip_counter
            info["contact_forces"] = observation["contact_forces"].copy()

        return observation, reward, terminated, truncated, info

    def render(self) -> Union[np.ndarray, None]:
        """Call the ``render`` method to update the renderer. It should be
        called every iteration; the method will decide by itself whether
        action is required.

        Returns
        -------
        np.ndarray
            The rendered image is one is rendered.
        """
        if self.render_mode == "headless":
            return None
        if self.curr_time < len(self._frames) * self._eff_render_interval:
            return None
        if self.render_mode == "saved":
            width, height = self.sim_params.render_window_size
            camera = self.sim_params.render_camera
            if self.update_camera_pos:
                self._update_cam_pos()
            if self.sim_params.camera_follows_fly_orientation:
                self._update_cam_rot()
            if self.sim_params.draw_adhesion:
                self._draw_adhesion()
            if self.sim_params.align_camera_with_gravity:
                self._rotate_camera()
            img = self.physics.render(width=width, height=height, camera_id=camera)
            img = img.copy()
            if self.sim_params.draw_contacts:
                img = self._draw_contacts(img)
            if self.sim_params.draw_gravity:
                img = self._draw_gravity(img)

            render_playspeed_text = self.sim_params.render_playspeed_text
            render_time_text = self.sim_params.render_timestamp_text
            if render_playspeed_text or render_time_text:
                if render_playspeed_text and render_time_text:
                    text = (
                        f"{self.curr_time:.2f}s ({self.sim_params.render_playspeed}x)"
                    )
                elif render_playspeed_text:
                    text = f"{self.sim_params.render_playspeed}x"
                elif render_time_text:
                    text = f"{self.curr_time:.2f}s"
                img = cv2.putText(
                    img,
                    text,
                    org=(20, 30),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=0.8,
                    color=(0, 0, 0),
                    lineType=cv2.LINE_AA,
                    thickness=1,
                )

            self._frames.append(img)
            self._last_render_time = self.curr_time
            return self._frames[-1]
        else:
            raise NotImplementedError

    def _update_cam_pos(self):
        cam = self.physics.bind(self._cam)
        cam_pos = cam.xpos.copy()
        cam_pos[2] = self.cam_offset[2] + self._floor_height
        cam.xpos = cam_pos

    def _update_cam_rot(self):
        cam = self.physics.bind(self._cam)
        cam_name = self._cam.name
        fly_z_rot_euler = np.array(
            [self.fly_rot[0], 0.0, 0.0]
            - self.spawn_orientation[::-1]
            - [np.pi / 2, 0, 0]
        )
        # This compensates both for the scipy to mujoco transform (align with y is
        # [0, 0, 0] in mujoco but [pi/2, 0, 0] in scipy) and the fact that the fly
        # orientation is already taken into account in the base_camera_rot (see below)
        # camera is always looking along its -z axis
        if cam_name in ["camera_top", "camera_bottom"]:
            # if camera is top or bottom always keep rotation around z only
            cam_matrix = R.from_euler("zyx", fly_z_rot_euler).as_matrix()
        elif cam_name in ["camera_front", "camera_back", "camera_left", "camera_right"]:
            # if camera is front, back, left or right apply the rotation around y
            cam_matrix = R.from_euler("yzx", fly_z_rot_euler).as_matrix()

        if cam_name in ["camera_bottom"]:
            cam_matrix = cam_matrix.T
            # z axis is inverted

        cam_matrix = self.base_camera_rot @ cam_matrix
        cam.xmat = cam_matrix.flatten()

    def _rotate_camera(self):
        # get camera
        cam = self.physics.bind(self._cam)
        # rotate the cam
        cam_matrix_base = getattr(cam, "xmat").copy()
        cam_matrix = self._camera_rot @ cam_matrix_base.reshape(3, 3)
        setattr(cam, "xmat", cam_matrix.flatten())

        return 0

    def _draw_adhesion(self):
        """Highlight the tarsal segments of the leg having adhesion"""
        if np.any(self._last_adhesion == 1):
            self.physics.named.model.geom_rgba[
                self._leg_adhesion_drawing_segments[self._last_adhesion == 1].ravel()
            ] = self._adhesion_rgba
        if np.any(self._active_adhesion):
            self.physics.named.model.geom_rgba[
                self._leg_adhesion_drawing_segments[self._active_adhesion].ravel()
            ] = self._active_adhesion_rgba
        if np.any(self._last_adhesion == 0):
            self.physics.named.model.geom_rgba[
                self._leg_adhesion_drawing_segments[self._last_adhesion == 0].ravel()
            ] = self._base_rgba
        return

    def _draw_gravity(self, img: np.ndarray) -> np.ndarray:
        """Draw gravity as an arrow. The arrow is drawn at the top right
        of the frame.
        """

        camera_matrix = self._dm_camera.matrix

        if self.sim_params.align_camera_with_gravity:
            arrow_start = self._last_fly_pos + self._camera_rot @ self._arrow_offset
        else:
            arrow_start = self._last_fly_pos + self._arrow_offset

        arrow_end = (
            arrow_start
            + self.physics.model.opt.gravity * self.sim_params.gravity_arrow_scaling
        )

        xyz_global = np.array([arrow_start, arrow_end]).T

        # Camera matrices multiply homogenous [x, y, z, 1] vectors.
        corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
        corners_homogeneous[:3, :] = xyz_global

        # Project world coordinates into pixel space. See:
        # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
        xs, ys, s = camera_matrix @ corners_homogeneous

        # x and y are in the pixel coordinate system.
        x = np.rint(xs / s).astype(int)
        y = np.rint(ys / s).astype(int)

        img = img.astype(np.uint8)
        img = cv2.arrowedLine(img, (x[0], y[0]), (x[1], y[1]), self._gravity_rgba, 10)

        return img

    def _draw_contacts(self, img: np.ndarray, thickness=2) -> np.ndarray:
        """Draw contacts as arrow which length is proportional to the force
        magnitude. The arrow is drawn at the center of the body. It uses
        the camera matrix to transfer from the global space to the pixels
        space.
        """

        def clip(p_in, p_out, z_clip):
            t = (z_clip - p_out[-1]) / (p_in[-1] - p_out[-1])
            return t * p_in + (1 - t) * p_out

        forces = self._last_contact_force
        pos = self._last_contact_pos

        magnitudes = np.linalg.norm(forces, axis=1)
        contact_indices = np.nonzero(magnitudes > self.sim_params.contact_threshold)[0]

        n_contacts = len(contact_indices)
        # Build an array of start and end points for the force arrows
        if n_contacts == 0:
            return img

        contact_forces = forces[contact_indices] * self.sim_params.force_arrow_scaling

        if self.sim_params.decompose_contacts:
            contact_pos = pos[:, None, contact_indices]
            Xw = contact_pos + (contact_forces[:, None] * _roll_eye).T
        else:
            contact_pos = pos[:, contact_indices]
            Xw = np.stack((contact_pos, contact_pos + contact_forces.T), 1)

        # Convert to homogeneous coordinates
        Xw = np.concatenate((Xw, np.ones((1, *Xw.shape[1:]))))

        mat = self._dm_camera.matrices()

        # Project to camera space
        Xc = np.tensordot(mat.rotation @ mat.translation, Xw, 1)
        Xc = Xc[:3, :] / Xc[-1, :]

        z_near = -self.physics.model.vis.map.znear * self.physics.model.stat.extent

        is_behind_cam = Xc[2] >= z_near
        is_visible = ~(is_behind_cam[0] & is_behind_cam[1:])

        is_out = is_visible & is_behind_cam[1:]
        is_in = np.where(is_visible & is_behind_cam[0])

        if self.sim_params.decompose_contacts:
            lines = np.stack((np.stack([Xc[:, 0]] * 3, axis=1), Xc[:, 1:]), axis=1)
        else:
            lines = Xc[:, :, None]

        lines[:, 1, is_out] = clip(lines[:, 0, is_out], lines[:, 1, is_out], z_near)
        lines[:, 0, is_in] = clip(lines[:, 1, is_in], lines[:, 0, is_in], z_near)

        # Project to pixel space
        lines = np.tensordot((mat.image @ mat.focal)[:, :3], lines, axes=1)
        lines2d = lines[:2] / lines[-1]
        lines2d = lines2d.T

        if not self.sim_params.perspective_arrow_length:
            unit_vectors = lines2d[:, :, 1] - lines2d[:, :, 0]
            length = np.linalg.norm(unit_vectors, axis=-1, keepdims=True)
            length[length == 0] = 1
            unit_vectors /= length
            lines2d[:, :, 1] = (
                lines2d[:, :, 0] + np.abs(contact_forces[:, :, None]) * unit_vectors
            )

        lines2d = np.rint(lines2d.reshape((-1, 2, 2))).astype(int)

        argsort = lines[2, 0].T.ravel().argsort()
        color_indices = np.tile(np.arange(3), lines.shape[-1])

        img = img.astype(np.uint8)

        for j in argsort:
            if not is_visible.ravel()[j]:
                continue

            color = self._decompose_colors[color_indices[j]]
            p1, p2 = lines2d[j]
            arrow_length = np.linalg.norm(p2 - p1)

            if arrow_length > 1e-2:
                r = self.sim_params.tip_length / arrow_length
            else:
                r = 1e-4

            if is_out.ravel()[j] and self.sim_params.perspective_arrow_length:
                cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
            else:
                cv2.arrowedLine(img, p1, p2, color, thickness, cv2.LINE_AA, tipLength=r)
        return img

    def _update_vision(self) -> None:
        """Check if the visual input needs to be updated (because the
        vision update freq does not necessarily match the physics
        simulation timestep). If needed, update the visual input of the fly
        and buffer it to ``self._curr_raw_visual_input``.
        """
        vision_config = self._config["vision"]
        next_render_time = (
            self._last_vision_update_time + self._eff_visual_render_interval
        )
        # avoid floating point errors: when too close, update anyway
        if self.curr_time + 0.5 * self.timestep < next_render_time:
            return
        raw_visual_input = []
        ommatidia_readouts = []
        for geom in self._geoms_to_hide:
            self.physics.named.model.geom_rgba[f"Animat/{geom}"] = [0.5, 0.5, 0.5, 0]
        self.arena.pre_visual_render_hook(self.physics)
        for side in ["L", "R"]:
            raw_img = self.physics.render(
                width=vision_config["raw_img_width_px"],
                height=vision_config["raw_img_height_px"],
                camera_id=f"Animat/{side}Eye_cam",
            )
            fish_img = np.ascontiguousarray(self.retina.correct_fisheye(raw_img))
            readouts_per_eye = self.retina.raw_image_to_hex_pxls(fish_img)
            ommatidia_readouts.append(readouts_per_eye)
            raw_visual_input.append(fish_img)
        for geom in self._geoms_to_hide:
            self.physics.named.model.geom_rgba[f"Animat/{geom}"] = [0.5, 0.5, 0.5, 1]
        self.arena.post_visual_render_hook(self.physics)
        self._curr_visual_input = np.array(ommatidia_readouts)
        if self.sim_params.render_raw_vision:
            self._curr_raw_visual_input = np.array(raw_visual_input)
        self._last_vision_update_time = self.curr_time

    def change_segment_color(self, segment, color):
        """Change the color of a segment of the fly.

        Parameters
        ----------
        segment : str
            The name of the segment to change the color of.
        color : Tuple[float, float, float, float]
            Target color as RGBA values normalized to [0, 1].
        """
        self.physics.named.model.geom_rgba[f"Animat/{segment}"] = color

    @property
    def vision_update_mask(self) -> np.ndarray:
        """
        The refresh frequency of the visual system is often loser than the
        same as the physics simulation time step. This 1D mask, whose
        size is the same as the number of simulation time steps, indicates
        in which time steps the visual inputs have been refreshed. In other
        words, the visual input frames where this mask is False are
        repetitions of the previous updated visual input frames.
        """
        return np.array(self._vision_update_mask)

    def get_observation(self) -> Tuple[ObsType, Dict[str, Any]]:
        """Get observation without stepping the physics simulation.

        Returns
        -------
        ObsType
            The observation as defined by the environment.
        """
        # joint sensors
        joint_obs = np.zeros((3, len(self.actuated_joints)))
        joint_sensordata = self.physics.bind(self._joint_sensors).sensordata
        for i, joint in enumerate(self.actuated_joints):
            base_idx = i * 5
            # pos and vel
            joint_obs[:2, i] = joint_sensordata[base_idx : base_idx + 2]
            # torque from pos/vel/motor actuators
            joint_obs[2, i] = joint_sensordata[base_idx + 2 : base_idx + 5].sum()
        joint_obs[2, :] *= 1e-9  # convert to N

        # fly position and orientation
        cart_pos = self.physics.bind(self._body_sensors[0]).sensordata
        cart_vel = self.physics.bind(self._body_sensors[1]).sensordata

        quat = self.physics.bind(self._body_sensors[2]).sensordata
        # ang_pos = transformations.quat_to_euler(quat)
        ang_pos = R.from_quat(quat[[1, 2, 3, 0]]).as_euler(
            "ZYX"
        )  # explicitly use extrinsic ZYX
        # ang_pos[0] *= -1  # flip roll??
        ang_vel = self.physics.bind(self._body_sensors[3]).sensordata
        fly_pos = np.array([cart_pos, cart_vel, ang_pos, ang_vel])

        if self.sim_params.camera_follows_fly_orientation:
            self.fly_rot = ang_pos

        if self.sim_params.draw_gravity:
            self._last_fly_pos = cart_pos

        # contact forces from crf_ext (first three components are rotational)
        contact_forces = self.physics.named.data.cfrc_ext[
            self.contact_sensor_placements
        ][:, 3:].copy()
        if self.sim_params.enable_adhesion:
            # Adhesion inputs force in the contact. Let's compute this force
            # and remove it from the contact forces
            contactid_normal = {}
            self._active_adhesion = np.zeros(self.n_legs, dtype=bool)
            for contact in self.physics.data.contact:
                id = np.where(self._adhesion_actuator_geomid == contact.geom1)
                if len(id[0]) > 0 and contact.exclude == 0:
                    contact_sensor_id = self._adhesion_bodies_with_contact_sensors[id][
                        0
                    ]
                    if contact_sensor_id in contactid_normal:
                        contactid_normal[contact_sensor_id].append(contact.frame[:3])
                    else:
                        contactid_normal[contact_sensor_id] = [contact.frame[:3]]
                    self._active_adhesion[id] = True
                id = np.where(self._adhesion_actuator_geomid == contact.geom2)
                if len(id[0]) > 0 and contact.exclude == 0:
                    contact_sensor_id = self._adhesion_bodies_with_contact_sensors[id][
                        0
                    ]
                    if contact_sensor_id in contactid_normal:
                        contactid_normal[contact_sensor_id].append(contact.frame[:3])
                    else:
                        contactid_normal[contact_sensor_id] = [contact.frame[:3]]
                    self._active_adhesion[id] = True

            for contact_sensor_id, normal in contactid_normal.items():
                adh_actuator_id = (
                    self._adhesion_bodies_with_contact_sensors == contact_sensor_id
                )
                if self._last_adhesion[adh_actuator_id] > 0:
                    if len(np.shape(normal)) > 1:
                        normal = np.mean(normal, axis=0)
                    contact_forces[contact_sensor_id, :] -= (
                        self.sim_params.adhesion_force * normal
                    )

        # if draw contacts same last contact forces and positions
        if self.sim_params.draw_contacts:
            self._last_contact_force = contact_forces
            self._last_contact_pos = (
                self.physics.named.data.xpos[self.contact_sensor_placements].copy().T
            )

        # end effector position
        ee_pos = self.physics.bind(self._end_effector_sensors).sensordata.copy()
        ee_pos = ee_pos.reshape((self.n_legs, 3))

        orientation_vec = self.physics.bind(self._body_sensors[4]).sensordata.copy()

        obs = {
            "joints": joint_obs.astype(np.float32),
            "fly": fly_pos.astype(np.float32),
            "contact_forces": contact_forces.astype(np.float32),
            "end_effectors": ee_pos.astype(np.float32),
            "fly_orientation": orientation_vec.astype(np.float32),
        }

        # olfaction
        if self.sim_params.enable_olfaction:
            antennae_pos = self.physics.bind(self._antennae_sensors).sensordata
            odor_intensity = self.arena.get_olfaction(antennae_pos.reshape(4, 3))
            obs["odor_intensity"] = odor_intensity.astype(np.float32)

        # vision
        if self.sim_params.enable_vision:
            self._update_vision()
            obs["vision"] = self._curr_visual_input.astype(np.float32)

        return obs

    def get_reward(self):
        """Get the reward for the current state of the environment. This
        method always returns 0 unless extended by the user.

        Returns
        -------
        float
            The reward.
        """
        return 0

    def is_terminated(self):
        """Whether the episode has terminated due to factors that are
        defined within the Markov Decision Process (e.g. task completion/
        failure, etc.). This method always returns False unless extended by
        the user.

        Returns
        -------
        bool
            Whether the simulation is terminated.
        """
        return False

    def is_truncated(self):
        """Whether the episode has terminated due to factors beyond the
            Markov Decision Process (e.g. time limit, etc.). This method
            always returns False unless extended by the user.

        Returns
        -------
        bool
            Whether the simulation is truncated.
        """
        return False

    def get_info(self):
        """Any additional information that is not part of the observation.
        This method always returns an empty dictionary unless extended by
        the user.

        Returns
        -------
        Dict[str, Any]
            The dictionary containing additional information.
        """
        info = {}
        if self.sim_params.enable_vision:
            if self.sim_params.render_raw_vision:
                info["raw_vision"] = self._curr_raw_visual_input.astype(np.float32)
        return info

    def save_video(self, path: Union[str, Path], stabilization_time=0.02):
        """Save rendered video since the beginning or the last ``reset()``,
        whichever is the latest. Only useful if ``render_mode`` is 'saved'.

        Parameters
        ----------
        path : str or Path
            Path to which the video should be saved.
        stabilization_time : float, optional
            Time (in seconds) to wait before starting to render the video.
            This might be wanted because it takes a few frames for the
            position controller to move the joints to the specified angles
            from the default, all-stretched position. By default 0.02s
        """
        if self.render_mode != "saved":
            logging.warning(
                'Render mode is not "saved"; no video will be saved despite '
                "`save_video()` call."
            )
        elif len(self._frames) == 0:
            logging.warning(
                "No frames have been rendered yet; no video will be saved despite "
                "`save_video()` call. Be sure to call `.render()` in your simulation "
                "loop."
            )

        num_stab_frames = int(np.ceil(stabilization_time / self._eff_render_interval))

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving video to {path}")
        with imageio.get_writer(path, fps=self.sim_params.render_fps) as writer:
            for frame in self._frames[num_stab_frames:]:
                writer.append_data(frame)

    def _get_center_of_mass(self):
        """Get the center of mass of the fly.
        (subtree com weighted by mass) STILL NEEDS TO BE TESTED MORE THOROUGHLY

        Returns
        -------
        np.ndarray
            The center of mass of the fly.
        """
        return np.average(
            self.physics.data.subtree_com, axis=0, weights=self.physics.data.crb[:, 0]
        )

    def _get_energy(self):
        """Get the energy of the system (kinetic, potential).

        Returns
        -------
        np.ndarray
            The energy of the system (kinetic, potential).
        """
        if not self.model.option.flag.energy == "enable":
            raise ValueError("Energy flag not activated in the mujoco xml file. ")
        return self.data.energy

    def close(self) -> None:
        """Close the environment, save data, and release any resources."""
        if self.render_mode == "saved" and self.output_dir is not None:
            self.save_video(self.output_dir / "video.mp4")

    def _stabilize_head(self):
        quat = self.physics.bind(self._thorax).xquat
        quat_inv = transformations.quat_inv(quat)
        roll, pitch, yaw = transformations.quat_to_euler(quat_inv, ordering="XYZ")
        self.physics.bind(self._head_stabilization_actuators).ctrl = roll, pitch