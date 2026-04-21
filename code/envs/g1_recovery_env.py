"""
G1 Recovery Environment for RecoverFormer.

A MuJoCo-based environment where the Unitree G1 humanoid receives random push
perturbations and must recover balance, optionally using environmental contacts
(walls, railings).

PD gains, action scale, observation scaling, and reward design are based on
unitree_rl_gym, unitree_rl_lab, unitree_rl_mjlab, and MuJoCo Playground.

Supports three environment variants:
  - "open_floor": flat ground, no contact surfaces
  - "walled": flat ground + walls at varying distances
  - "cluttered": flat ground + walls + railings + table surfaces
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Optional


# Path to the G1 MuJoCo model
_MENAGERIE_DIR = Path(__file__).resolve().parent.parent.parent / "mujoco_menagerie"
_G1_XML = str(_MENAGERIE_DIR / "unitree_g1" / "scene.xml")

# Contact region definitions (K_c = 8)
CONTACT_REGION_NAMES = [
    "wall_front_left", "wall_front_right",
    "wall_back_left", "wall_back_right",
    "railing_left", "railing_right",
    "table_left", "table_right",
]
K_C = len(CONTACT_REGION_NAMES)

N_ACTUATORS = 29

# Arm joint indices (for detecting hand contacts)
LEFT_WRIST_JOINTS = [20, 21, 22]
RIGHT_WRIST_JOINTS = [27, 28, 29]

# Observation dimensions
# joint_pos(29) + joint_vel(29) + projected_gravity(3) + torso_angvel(3) + torso_linvel(3)
# + foot_contacts(2) + contact_region_distances(8) + prev_action(29) = 106
OBS_DIM_BASE = 106

# Action scale: 0.25 radians (standard across all Unitree repos)
# NOTE: The G1 scene.xml defines position actuators with built-in PD control
# (kp=500, per-joint kd). We set data.ctrl to joint position targets directly.
ACTION_SCALE = 0.25


class G1RecoveryEnv(gym.Env):
    """Gymnasium environment for G1 humanoid push recovery."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        env_type: str = "open_floor",
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        control_freq: float = 50.0,
        # Perturbation settings
        push_force_range: tuple = (50.0, 300.0),
        push_interval_range: tuple = (1.0, 3.0),
        # Domain randomization
        friction_range: tuple = (0.5, 1.2),
        mass_perturbation: float = 0.0,
        latency_steps: int = 0,
        latency_steps_range: Optional[tuple] = None,  # (min, max); None uses fixed latency_steps
        # Wall placement (walled env only); None = random in [0.4, 0.8]
        wall_distance: Optional[float] = None,
        # Reward weights
        reward_weights: Optional[dict] = None,
    ):
        super().__init__()

        self.env_type = env_type
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.control_freq = control_freq

        self.push_force_range = push_force_range
        self.push_interval_range = push_interval_range

        self.friction_range = friction_range
        self.mass_perturbation = mass_perturbation
        self.latency_steps = latency_steps
        self.latency_steps_range = latency_steps_range
        self.wall_distance = wall_distance
        self._action_delay_buf = []  # FIFO queue for delayed action application


        # ── Reward weights (based on unitree_rl_gym / unitree_rl_mjlab) ──
        self.reward_w = {
            # Positive (exp-kernel tracking terms, bounded in [0, weight])
            "orientation": 5.0,       # upright: -sum(projected_gravity_xy^2), from unitree_rl_lab
            "base_height": 10.0,      # (h - 0.78)^2, from unitree_rl_lab
            "default_pose": 1.0,      # exp kernel: stay near default joint angles
            "alive": 0.15,            # small constant, from unitree_rl_gym
            "feet_contact": 0.5,      # both feet on ground
            "com_over_feet": 2.0,     # exp kernel: COM above support center
            # Penalties
            "termination": 200.0,     # large terminal penalty, from unitree_rl_mjlab
            "ang_vel_xy": 0.05,       # from unitree_rl_gym
            "lin_vel_z": 2.0,         # from unitree_rl_gym
            "joint_vel": 1e-3,        # from unitree_rl_gym
            "action_rate": 0.05,      # from unitree_rl_lab
            "torque": 2e-5,           # from unitree_rl_lab (energy)
            "joint_acc": 2.5e-7,      # from unitree_rl_gym
            "joint_pos_limits": 5.0,  # near joint limit penalty, from unitree_rl_gym
            # Contact rewards (walled/cluttered only)
            "useful_contact": 5.0,
            "harmful_contact": 10.0,
        }
        if reward_weights is not None:
            self.reward_w.update(reward_weights)

        self._load_model()

        obs_dim = OBS_DIM_BASE
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_ACTUATORS,), dtype=np.float32
        )

        # State tracking
        self._step_count = 0
        self._prev_action = np.zeros(N_ACTUATORS, dtype=np.float32)
        self._prev_joint_vel = np.zeros(N_ACTUATORS, dtype=np.float64)
        self._next_push_time = 0.0
        self._is_pushed = False
        self._push_applied = False
        self._nominal_height = 0.78
        self._terminated = False

        self._contact_surfaces = []

        self._renderer = None
        if render_mode in ("human", "rgb_array"):
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    def _load_model(self):
        """Load the G1 MuJoCo model, optionally adding contact surfaces."""
        if self.env_type == "open_floor":
            self.model = mujoco.MjModel.from_xml_path(_G1_XML)
        else:
            xml_str = self._build_env_xml()
            # from_xml_string resolves <include> paths relative to CWD;
            # temporarily switch to the g1 directory so the includes resolve.
            import os
            _orig_cwd = os.getcwd()
            os.chdir(Path(_G1_XML).parent)
            try:
                self.model = mujoco.MjModel.from_xml_string(xml_str)
            finally:
                os.chdir(_orig_cwd)

        self.data = mujoco.MjData(self.model)
        self._physics_dt = self.model.opt.timestep
        self._n_substeps = int(1.0 / (self.control_freq * self._physics_dt))

        mujoco.mj_resetData(self.model, self.data)

        # Use keyframe "stand" as the default pose if available
        if self.model.nkey > 0:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
            if key_id >= 0:
                self.data.qpos[:] = self.model.key_qpos[key_id]

        mujoco.mj_forward(self.model, self.data)
        self._default_qpos = self.data.qpos.copy()
        self._default_qvel = self.data.qvel.copy()

        # Body IDs
        self._torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self._left_foot_id = self._find_body("left_ankle_roll_link")
        self._right_foot_id = self._find_body("right_ankle_roll_link")
        self._left_hand_id = self._find_body("left_wrist_yaw_link")
        self._right_hand_id = self._find_body("right_wrist_yaw_link")

        # Cache joint ranges for limit penalty
        self._joint_ranges = np.zeros((N_ACTUATORS, 2))
        for i in range(N_ACTUATORS):
            jnt_id = i + 1  # skip floating base joint
            if jnt_id < self.model.njnt:
                self._joint_ranges[i] = self.model.jnt_range[jnt_id]

    def _find_body(self, name):
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        except Exception:
            return -1

    def _build_env_xml(self):
        with open(_G1_XML, "r") as f:
            original_xml = f.read()

        objects_xml = ""

        if self.env_type in ("walled", "cluttered"):
            wall_dist = self.wall_distance if self.wall_distance is not None else np.random.uniform(0.4, 0.8)
            objects_xml += f"""
            <body name="wall_front" pos="{wall_dist} 0 0.5">
                <geom name="wall_front_geom" type="box" size="0.02 1.0 0.5"
                       rgba="0.7 0.7 0.8 0.8" friction="0.8 0.005 0.001"/>
            </body>
            <body name="wall_left" pos="0 {wall_dist} 0.5">
                <geom name="wall_left_geom" type="box" size="1.0 0.02 0.5"
                       rgba="0.7 0.7 0.8 0.8" friction="0.8 0.005 0.001"/>
            </body>
            """

        if self.env_type == "cluttered":
            railing_dist = np.random.uniform(0.3, 0.6)
            objects_xml += f"""
            <body name="railing_right" pos="0 -{railing_dist} 0.9">
                <geom name="railing_geom" type="cylinder" size="0.025 0.8"
                       rgba="0.5 0.5 0.5 1.0" friction="0.9 0.005 0.001"
                       euler="0 1.5708 0"/>
            </body>
            <body name="table_front" pos="{np.random.uniform(0.4, 0.7)} 0 0.4">
                <geom name="table_geom" type="box" size="0.3 0.4 0.02"
                       rgba="0.6 0.4 0.2 1.0" friction="0.7 0.005 0.001"/>
            </body>
            """

        if objects_xml:
            return original_xml.replace("</worldbody>", objects_xml + "\n</worldbody>")
        return original_xml

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._default_qpos
        self.data.qvel[:] = 0.0

        # Set ctrl to default position targets so initial step is stable
        self.data.ctrl[:N_ACTUATORS] = self._default_qpos[7:]

        # Small random perturbation to initial joint angles
        self.data.qpos[7:] += np.random.uniform(-0.02, 0.02, size=self.model.nq - 7)

        # Domain randomization
        if self.mass_perturbation > 0:
            for i in range(self.model.nbody):
                orig = self.model.body_mass[i]
                if orig > 0.01:
                    scale = 1.0 + np.random.uniform(-self.mass_perturbation, self.mass_perturbation)
                    self.model.body_mass[i] = orig * scale

        if self.friction_range != (1.0, 1.0):
            for i in range(self.model.ngeom):
                fscale = np.random.uniform(*self.friction_range)
                self.model.geom_friction[i, 0] = max(0.1, self.model.geom_friction[i, 0] * fscale)

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_action = np.zeros(N_ACTUATORS, dtype=np.float32)
        self._prev_joint_vel = np.zeros(N_ACTUATORS, dtype=np.float64)
        self._next_push_time = np.random.uniform(*self.push_interval_range)
        self._is_pushed = False
        self._push_applied = False
        self._terminated = False

        # Per-episode latency: sample from range if provided, else use fixed value
        if self.latency_steps_range is not None:
            lo, hi = self.latency_steps_range
            self._episode_latency = np.random.randint(lo, hi + 1)
        else:
            self._episode_latency = self.latency_steps
        self._action_delay_buf = []

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Action latency: buffer actions and apply a delayed one
        if self._episode_latency > 0:
            self._action_delay_buf.append(action)
            if len(self._action_delay_buf) > self._episode_latency:
                applied_action = self._action_delay_buf.pop(0)
            else:
                applied_action = np.zeros_like(action)  # default pose while buffer fills
        else:
            applied_action = action

        # Action = position offset from default pose (0.25 rad scale)
        # G1 scene.xml actuators are position-controlled with built-in PD (kp=500)
        joint_targets = self._default_qpos[7:] + applied_action * ACTION_SCALE

        # Push perturbation
        current_time = self.data.time
        if not self._push_applied and current_time >= self._next_push_time:
            self._apply_push()
            self._push_applied = True
            self._is_pushed = True

        # Set position targets and step physics
        self.data.ctrl[:N_ACTUATORS] = joint_targets
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward, reward_info = self._compute_reward(action)
        terminated = self._check_termination()
        truncated = self._step_count >= self.max_episode_steps

        if terminated:
            reward -= self.reward_w["termination"]
            reward_info["r_termination"] = -self.reward_w["termination"]

        _, _, useful_left, useful_right = self._get_hand_contacts()
        info = {
            "reward_info": reward_info,
            "is_pushed": self._is_pushed,
            "com_height": self.data.qpos[2],
            "torso_tilt": self._get_torso_tilt(),
            "com_vel": np.linalg.norm(self.data.qvel[:3]),
            "useful_contact": useful_left or useful_right,
        }
        self._prev_action = action.astype(np.float32)

        return obs, reward, terminated, truncated, info

    def _apply_push(self):
        """Apply instantaneous velocity impulse (standard IsaacLab approach)."""
        force_mag = np.random.uniform(*self.push_force_range)
        angle = np.random.uniform(0, 2 * np.pi)
        force = np.array([
            force_mag * np.cos(angle),
            force_mag * np.sin(angle),
            np.random.uniform(-force_mag * 0.1, force_mag * 0.2),
        ])
        impulse_duration = 0.1
        total_mass = max(np.sum(self.model.body_mass), 1.0)
        self.data.qvel[0:3] += force * impulse_duration / total_mass

    def _get_obs(self):
        """Observation vector following unitree_rl_gym conventions."""
        # Joint positions relative to default (29)
        joint_pos = (self.data.qpos[7:] - self._default_qpos[7:]).astype(np.float32)

        # Joint velocities, scaled by 0.05 (29)
        joint_vel = self.data.qvel[6:].astype(np.float32) * 0.05

        # Projected gravity (3) — replaces quaternion, more informative
        proj_grav = self._get_projected_gravity().astype(np.float32)

        # Torso angular velocity, scaled by 0.25 (3)
        torso_angvel = self.data.qvel[3:6].astype(np.float32) * 0.25

        # Torso linear velocity (3)
        torso_linvel = self.data.qvel[0:3].astype(np.float32)

        # Foot contacts (2)
        foot_contacts = np.zeros(2, dtype=np.float32)
        if self._left_foot_id >= 0:
            foot_contacts[0] = float(self._check_body_contact(self._left_foot_id))
        if self._right_foot_id >= 0:
            foot_contacts[1] = float(self._check_body_contact(self._right_foot_id))

        # Contact region distances (8)
        contact_dists = self._get_contact_region_distances()

        # Previous action (29)
        prev_action = self._prev_action

        parts = [
            joint_pos,       # 29
            joint_vel,       # 29
            proj_grav,       # 3  (was 4 for quat, now 3)
            torso_angvel,    # 3
            torso_linvel,    # 3
            foot_contacts,   # 2
            contact_dists,   # 8
            prev_action,     # 29
        ]  # total = 106
        obs = np.concatenate(parts)
        return obs.astype(np.float32)

    def _check_body_contact(self, body_id):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = self.model.geom_bodyid[c.geom1]
            g2 = self.model.geom_bodyid[c.geom2]
            if body_id in (g1, g2):
                return True
        return False

    def _get_hand_contacts(self):
        left_contact = right_contact = False
        useful_left = useful_right = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = self.model.geom_bodyid[c.geom1]
            g2 = self.model.geom_bodyid[c.geom2]
            if self._left_hand_id >= 0 and self._left_hand_id in (g1, g2):
                left_contact = True
                other = g2 if g1 == self._left_hand_id else g1
                if other != 0:
                    useful_left = True
            if self._right_hand_id >= 0 and self._right_hand_id in (g1, g2):
                right_contact = True
                other = g2 if g1 == self._right_hand_id else g1
                if other != 0:
                    useful_right = True
        return left_contact, right_contact, useful_left, useful_right

    def _get_contact_region_distances(self):
        distances = np.ones(K_C, dtype=np.float32) * 5.0
        if self.env_type == "open_floor":
            return distances
        lh = self.data.xpos[self._left_hand_id] if self._left_hand_id >= 0 else np.zeros(3)
        rh = self.data.xpos[self._right_hand_id] if self._right_hand_id >= 0 else np.zeros(3)
        for i, rn in enumerate(CONTACT_REGION_NAMES):
            gn = rn + "_geom" if "geom" not in rn else rn
            try:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gn)
                if gid >= 0:
                    gp = self.data.geom_xpos[gid]
                    distances[i] = min(np.linalg.norm(lh - gp), np.linalg.norm(rh - gp))
            except Exception:
                pass
        return distances / 2.0

    def _get_support_center_xy(self):
        lp = self.data.xpos[self._left_foot_id] if self._left_foot_id >= 0 else np.zeros(3)
        rp = self.data.xpos[self._right_foot_id] if self._right_foot_id >= 0 else np.zeros(3)
        return 0.5 * (lp[:2] + rp[:2])

    def _get_projected_gravity(self):
        if self._torso_body_id >= 0:
            rot = self.data.xmat[self._torso_body_id].reshape(3, 3)
            return rot.T @ np.array([0.0, 0.0, -1.0])
        return np.array([0.0, 0.0, -1.0])

    def _get_torso_tilt(self):
        if self._torso_body_id >= 0:
            rot = self.data.xmat[self._torso_body_id].reshape(3, 3)
            return np.arccos(np.clip(rot[2, 2], -1.0, 1.0))
        return 0.0

    def _compute_reward(self, action):
        """
        Reward function based on unitree_rl_gym / unitree_rl_lab / unitree_rl_mjlab.

        Positive terms use exponential kernels (bounded, smooth gradients).
        Penalties use L2 with small weights (don't dominate positive signal).
        Large termination penalty applied separately in step().
        """
        w = self.reward_w
        info = {}

        # ── Positive rewards ──

        # 1. Orientation: penalize projected gravity xy (from unitree_rl_lab, weight=-5.0)
        #    Reformulated as positive: high reward when upright
        proj_grav = self._get_projected_gravity()
        grav_xy_sq = proj_grav[0] ** 2 + proj_grav[1] ** 2
        # unitree_rl_lab uses: -5.0 * sum(grav_xy^2)
        # We use exp kernel for positive framing:
        r_orient = w["orientation"] * np.exp(-grav_xy_sq / 0.01)
        info["r_orient"] = r_orient

        # 2. Base height tracking (from unitree_rl_lab, weight=-10.0)
        com_h = self.data.qpos[2]
        h_err_sq = (com_h - self._nominal_height) ** 2
        r_height = w["base_height"] * np.exp(-h_err_sq / 0.005)
        info["r_height"] = r_height

        # 3. Default pose: reward staying near default joint angles
        #    (from unitree_rl_gym "hip_pos" and unitree_rl_mjlab "pose")
        joint_pos_err = self.data.qpos[7:] - self._default_qpos[7:]
        r_pose = w["default_pose"] * np.exp(-2.0 * np.mean(joint_pos_err ** 2))
        info["r_pose"] = r_pose

        # 4. Alive bonus (from unitree_rl_gym, weight=0.15)
        r_alive = w["alive"]
        info["r_alive"] = r_alive

        # 5. Feet contact
        lc = float(self._check_body_contact(self._left_foot_id)) if self._left_foot_id >= 0 else 0.0
        rc = float(self._check_body_contact(self._right_foot_id)) if self._right_foot_id >= 0 else 0.0
        r_feet = w["feet_contact"] * (lc + rc) * 0.5
        info["r_feet"] = r_feet

        # 6. COM over support center
        com_xy = self.data.qpos[0:2]
        support_xy = self._get_support_center_xy()
        com_dist_sq = np.sum((com_xy - support_xy) ** 2)
        r_com = w["com_over_feet"] * np.exp(-com_dist_sq / 0.02)
        info["r_com"] = r_com

        # ── Penalties (L2, small weights) ──

        # 7. Angular velocity xy (penalize rotational oscillation)
        ang_vel_xy = self.data.qvel[3:5]
        r_ang = -w["ang_vel_xy"] * np.sum(ang_vel_xy ** 2)
        info["r_ang"] = r_ang

        # 10. Linear velocity z (from unitree_rl_gym, weight=-2.0)
        r_linz = -w["lin_vel_z"] * self.data.qvel[2] ** 2
        info["r_linz"] = r_linz

        # 9. Joint velocity (from unitree_rl_gym, weight=-1e-3)
        jvel = self.data.qvel[6:]
        r_jvel = -w["joint_vel"] * np.sum(jvel ** 2)
        info["r_jvel"] = r_jvel

        # 10. Action rate (from unitree_rl_lab, weight=-0.05)
        r_arate = -w["action_rate"] * np.sum((action - self._prev_action) ** 2)
        info["r_arate"] = r_arate

        # 11. Torque / energy (from unitree_rl_lab, weight=-2e-5)
        actuator_forces = self.data.actuator_force[:N_ACTUATORS]
        r_torque = -w["torque"] * np.sum(actuator_forces ** 2)
        info["r_torque"] = r_torque

        # 12. Joint acceleration (from unitree_rl_gym, weight=-2.5e-7)
        jacc = (jvel - self._prev_joint_vel) * self.control_freq
        r_jacc = -w["joint_acc"] * np.sum(jacc ** 2)
        self._prev_joint_vel = jvel.copy()
        info["r_jacc"] = r_jacc

        # 13. Joint position limits (from unitree_rl_gym, weight=-5.0)
        #     Penalize when within 10% of joint limits
        qpos_joints = self.data.qpos[7:]
        soft_lower = self._joint_ranges[:, 0] + 0.1 * (self._joint_ranges[:, 1] - self._joint_ranges[:, 0])
        soft_upper = self._joint_ranges[:, 1] - 0.1 * (self._joint_ranges[:, 1] - self._joint_ranges[:, 0])
        below = np.clip(soft_lower - qpos_joints, 0, None)
        above = np.clip(qpos_joints - soft_upper, 0, None)
        r_jlim = -w["joint_pos_limits"] * np.sum(below ** 2 + above ** 2)
        info["r_jlim"] = r_jlim

        # 14. Contact rewards (walled/cluttered)
        r_contact = 0.0
        if self.env_type != "open_floor":
            _, _, ul, ur = self._get_hand_contacts()
            if ul or ur:
                r_contact = w["useful_contact"]
            if self._check_body_contact(self._torso_body_id):
                r_contact -= w["harmful_contact"]
        info["r_contact"] = r_contact

        total = (r_orient + r_height + r_pose + r_alive + r_feet + r_com
                 + r_ang + r_linz + r_jvel + r_arate + r_torque + r_jacc + r_jlim
                 + r_contact)
        info["total"] = total
        return float(total), info

    def _check_termination(self):
        if self.data.qpos[2] < 0.3:
            self._terminated = True
            return True
        if self._get_torso_tilt() > 1.2:
            self._terminated = True
            return True
        return False

    def render(self):
        if self._renderer is not None:
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
