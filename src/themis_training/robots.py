"""Per-robot profiles for the MPC-guided env builders.

Everything robot-specific that :mod:`themis_training.env_cfgs` needs — body /
site / geom names, joint groups, pose-reward stds, centroidal-MPC model
parameters — lives in one :class:`RobotProfile`, so the same builders produce
the THEMIS and the Unitree G1 tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.entity import EntityCfg
from mjlab.sensor import ContactMatch

from themis_training.g1 import g1_constants as g1
from themis_training.themis import themis_constants as themis


@dataclass(frozen=True)
class SelfCollisionTerm:
  """A penalised self-collision: one contact sensor + one reward term."""
  name: str
  primary: ContactMatch
  secondary: ContactMatch
  force_threshold: float


@dataclass(frozen=True)
class RobotProfile:
  name: str
  robot_cfg_fn: Callable[..., EntityCfg]
  joint_names: tuple[str, ...]
  # Scalar, or per-joint-regex dict (``JointPositionActionCfg.scale``).
  action_scale: float | dict[str, float]

  # ── Bodies ──
  root_body: str        # floating-base body (raycast frame, CoM/inertia DR)
  upright_body: str     # upright / body-ang-vel rewards, viewer
  command_viz_z: float  # twist-arrow height above the root

  # ── Feet ──
  foot_sites: tuple[str, str]           # (left, right) sole sites
  foot_friction_geoms: tuple[str, ...]  # geoms randomised by foot_friction
  # Per-foot ground-contact matches (left, right).  ``feet_ground_contact``
  # concatenates them so its columns are [left, right].
  left_foot_contact: ContactMatch
  right_foot_contact: ContactMatch
  # Heel / toe geoms for ``cop_forward`` (THEMIS has split sole geoms).  When
  # ``None``, ``cop_foot_geoms`` is used with the contact-position CoP reward.
  heel_geoms: tuple[str, ...] | None
  toe_geoms: tuple[str, ...] | None
  cop_foot_geoms: tuple[tuple[str, ...], tuple[str, ...]] | None
  # Sole extents w.r.t. the foot site (centroidal-MPC CoP constraints and the
  # contact-position CoP reward).
  foot_x_toe: float
  foot_x_heel: float
  foot_y_half: float
  swing_height: float  # foot_clearance target (m)

  # ── Contacts ──
  self_collisions: tuple[SelfCollisionTerm, ...]
  nonfoot_ground_geoms: tuple[str, ...]  # illegal-contact termination
  leg_geoms: tuple[str, ...]             # leg-box collision penalty
  body_box_geoms: tuple[str, ...]        # non-hand robot-box net force (MPC)

  # ── Joint groups / pose reward ──
  pose_std_standing: dict[str, float]
  pose_std_walking: dict[str, float]
  pose_std_running: dict[str, float]
  # v2 ankle tightening (applied to std_walking / std_running).
  ankle_std_walking: dict[str, float]
  ankle_std_running: dict[str, float]
  # Arm groups for the loco-manip pose tightening: regex → std.
  arm_std_loco_manip: dict[str, float]
  arm_joint_exprs: tuple[str, ...]  # arm_joint_deviation termination
  arm_max_deviation: float          # its threshold (rad)

  # ── Centroidal MPC model ──
  mass: float
  inertia: tuple[tuple[float, ...], ...]  # whole-body inertia about CoM
  com_height: float
  hip_width: float
  loco_manip_hip_width: float  # hip_width for the loco-manipulation MPC
  gait_period: float

  # ── Sim (None = keep mjlab default) ──
  ccd_iterations: int | None = None
  contact_sensor_maxmatch: int | None = None
  nconmax: int | None = None

  experiment_name: str = ""


def _match_geoms(pattern: str) -> ContactMatch:
  return ContactMatch(mode="geom", pattern=pattern, entity="robot")


def _match_bodies(pattern: str) -> ContactMatch:
  return ContactMatch(mode="body", pattern=pattern, entity="robot")


def _self_collision(name: str, a: ContactMatch, b: ContactMatch, f: float) -> SelfCollisionTerm:
  return SelfCollisionTerm(name=name, primary=a, secondary=b, force_threshold=f)


# ══════════════════════════════════════════════════════════════════════════════
# THEMIS TH02-A7
# ══════════════════════════════════════════════════════════════════════════════

THEMIS = RobotProfile(
  name="Themis",
  robot_cfg_fn=themis.get_themis_robot_cfg,
  joint_names=tuple(themis.JOINT_NAMES_EXPR),
  action_scale=0.5,
  root_body="BASE_LINK",
  upright_body="BASE_LINK",
  command_viz_z=1.45,  # above the head (~1.18 m standing + margin)
  foot_sites=("left_foot", "right_foot"),
  foot_friction_geoms=("left_foot_collision", "right_foot_collision"),
  left_foot_contact=_match_geoms("left_foot_collision"),
  right_foot_contact=_match_geoms("right_foot_collision"),
  heel_geoms=(
    "FOOT_L_heel_in", "FOOT_L_heel_out", "FOOT_L_heel_center",
    "FOOT_R_heel_in", "FOOT_R_heel_out", "FOOT_R_heel_center",
  ),
  toe_geoms=(
    "FOOT_L_toe_in", "FOOT_L_toe_out", "FOOT_L_toe_in2", "FOOT_L_toe_out2",
    "FOOT_R_toe_in", "FOOT_R_toe_out", "FOOT_R_toe_in2", "FOOT_R_toe_out2",
  ),
  cop_foot_geoms=None,
  foot_x_toe=0.12,
  foot_x_heel=0.06,
  foot_y_half=0.04,
  swing_height=0.15,
  self_collisions=(
    # Hip abad links rubbing against each other.
    _self_collision("hip_abad_collision",
                    _match_bodies("HIP_ABAD_L"), _match_bodies("HIP_ABAD_R"), 10.0),
    # Feet touching each other.
    _self_collision("feet_feet_collision",
                    _match_geoms("FOOT_L_.*"), _match_geoms("FOOT_R_.*"), 5.0),
    # Cross-leg kicks during swing.
    _self_collision("right_foot_left_tibia_collision",
                    _match_geoms("FOOT_R_.*"), _match_geoms("left_tibia_collision"), 5.0),
    _self_collision("left_foot_right_tibia_collision",
                    _match_geoms("FOOT_L_.*"), _match_geoms("right_tibia_collision"), 5.0),
    # Hand slapping own thigh.
    _self_collision("left_femur_wrist_collision",
                    _match_bodies("FEMUR_L"), _match_bodies("UPPERWRIST_L"), 5.0),
    _self_collision("right_femur_wrist_collision",
                    _match_bodies("FEMUR_R"), _match_bodies("UPPERWRIST_R"), 5.0),
  ),
  nonfoot_ground_geoms=(
    "trunk_collision",
    "left_tibia_collision",
    "right_tibia_collision",
  ),
  leg_geoms=(
    "left_tibia_collision",
    "right_tibia_collision",
    "left_foot_collision",
    "right_foot_collision",
  ),
  body_box_geoms=(
    "left_tibia_collision",
    "right_tibia_collision",
    "left_foot_collision",
    "right_foot_collision",
    "left_upperarm_collision",
    "right_upperarm_collision",
  ),
  pose_std_standing={
    # Lower body
    r".*HIP_PITCH.*":  0.025,
    r".*HIP_ROLL.*":   0.15,
    r".*HIP_YAW.*":    0.15,
    r".*KNEE.*":       0.05,
    r".*ANKLE.*":      0.05,
    # Upper body
    r".*SHOULDER.*":   0.05,
    r".*ELBOW.*":      0.05,
    r".*WRIST.*":      0.1,
    r".*HEAD.*":       0.05,
  },
  pose_std_walking={
    # Lower body
    r".*HIP_PITCH.*":  0.1,
    r".*HIP_ROLL.*":   0.15,
    r".*HIP_YAW.*":    0.15,
    r".*KNEE.*":       0.35,
    r".*ANKLE.*":      0.25,
    # Upper body – arms swing during walking
    r".*SHOULDER.*":   0.2,
    r".*ELBOW.*":      0.2,
    r".*WRIST.*":      0.3,
    r".*HEAD.*":       0.1,
  },
  pose_std_running={
    # Lower body
    r".*HIP_PITCH.*":  0.25,
    r".*HIP_ROLL.*":   0.25,
    r".*HIP_YAW.*":    0.25,
    r".*KNEE.*":       0.6,
    r".*ANKLE.*":      0.35,
    # Upper body
    r".*SHOULDER.*":   0.35,
    r".*ELBOW.*":      0.35,
    r".*WRIST.*":      0.4,
    r".*HEAD.*":       0.15,
  },
  ankle_std_walking={r".*ANKLE.*": 0.15},
  ankle_std_running={r".*ANKLE.*": 0.20},
  arm_std_loco_manip={
    r".*SHOULDER.*": 0.05,
    r".*ELBOW.*":    0.05,
    r".*WRIST.*":    0.10,
  },
  arm_joint_exprs=(r".*SHOULDER.*", r".*ELBOW.*", r".*WRIST.*", r".*HEAD.*"),
  arm_max_deviation=0.25,
  mass=37.0,
  inertia=(
    ( 6.153, 0.0,   0.338),
    ( 0.0,   6.181, 0.0  ),
    ( 0.338, 0.0,   0.849),
  ),
  com_height=1.17,
  hip_width=0.15,
  loco_manip_hip_width=0.1,  # LocoMPCCommandCfg default, as originally trained
  gait_period=0.9,
  experiment_name="themis_velocity",
)


# ══════════════════════════════════════════════════════════════════════════════
# Unitree G1 (29 DOF)
# ══════════════════════════════════════════════════════════════════════════════

_G1_FOOT_GEOMS = tuple(
  tuple(f"{side}_foot{i}_collision" for i in range(1, 8)) for side in ("left", "right")
)

G1 = RobotProfile(
  name="G1",
  robot_cfg_fn=g1.get_g1_robot_cfg,
  joint_names=tuple(g1.JOINT_NAMES_EXPR),
  action_scale=g1.G1_ACTION_SCALE,
  root_body="pelvis",
  upright_body="torso_link",
  command_viz_z=1.15,
  foot_sites=("left_foot", "right_foot"),
  foot_friction_geoms=_G1_FOOT_GEOMS[0] + _G1_FOOT_GEOMS[1],
  # Each sole is 7 capsules; match the ankle-roll subtree so each foot is a
  # single sensor column.
  left_foot_contact=ContactMatch(mode="subtree", pattern="left_ankle_roll_link", entity="robot"),
  right_foot_contact=ContactMatch(mode="subtree", pattern="right_ankle_roll_link", entity="robot"),
  # The G1 capsules run heel-to-toe, so heel/toe force can't be split by
  # geom; cop_forward uses the contact positions instead.
  heel_geoms=None,
  toe_geoms=None,
  cop_foot_geoms=_G1_FOOT_GEOMS,
  # Capsules span x ∈ [-0.094, 0.092] around the sole site, |y| ≤ 0.037.
  # Slightly inset for the MPC CoP constraint.
  foot_x_toe=0.09,
  foot_x_heel=0.08,
  foot_y_half=0.03,
  swing_height=0.10,
  self_collisions=(
    # Any robot-robot contact (all G1 collision geoms collide with each other).
    _self_collision(
      "self_collision",
      ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      10.0,
    ),
  ),
  nonfoot_ground_geoms=(
    "pelvis_collision",
    "torso_collision",
    "head_collision",
    "left_shin_collision",
    "right_shin_collision",
  ),
  leg_geoms=(
    "left_shin_collision",
    "right_shin_collision",
  ) + _G1_FOOT_GEOMS[0] + _G1_FOOT_GEOMS[1],
  body_box_geoms=(
    "left_thigh_collision",
    "right_thigh_collision",
    "left_shin_collision",
    "right_shin_collision",
    "torso_collision",
    "left_shoulder_yaw_collision",
    "right_shoulder_yaw_collision",
    "left_elbow_yaw_collision",
    "right_elbow_yaw_collision",
  ) + _G1_FOOT_GEOMS[0] + _G1_FOOT_GEOMS[1],
  # THEMIS group stds mapped onto G1 joints; waist stds from mjlab's G1 task.
  pose_std_standing={
    r".*hip_pitch.*":   0.025,
    r".*hip_roll.*":    0.15,
    r".*hip_yaw.*":     0.15,
    r".*knee.*":        0.05,
    r".*ankle.*":       0.05,
    r".*waist.*":       0.05,
    r".*shoulder.*":    0.05,
    r".*elbow.*":       0.05,
    r".*wrist.*":       0.1,
  },
  pose_std_walking={
    r".*hip_pitch.*":   0.1,
    r".*hip_roll.*":    0.15,
    r".*hip_yaw.*":     0.15,
    r".*knee.*":        0.35,
    r".*ankle.*":       0.25,
    r".*waist_yaw.*":   0.2,
    r".*waist_roll.*":  0.08,
    r".*waist_pitch.*": 0.1,
    r".*shoulder.*":    0.2,
    r".*elbow.*":       0.2,
    r".*wrist.*":       0.3,
  },
  pose_std_running={
    r".*hip_pitch.*":   0.25,
    r".*hip_roll.*":    0.25,
    r".*hip_yaw.*":     0.25,
    r".*knee.*":        0.6,
    r".*ankle.*":       0.35,
    r".*waist_yaw.*":   0.3,
    r".*waist_roll.*":  0.08,
    r".*waist_pitch.*": 0.2,
    r".*shoulder.*":    0.35,
    r".*elbow.*":       0.35,
    r".*wrist.*":       0.4,
  },
  ankle_std_walking={r".*ankle.*": 0.15},
  ankle_std_running={r".*ankle.*": 0.20},
  arm_std_loco_manip={
    r".*shoulder.*": 0.05,
    r".*elbow.*":    0.05,
    r".*wrist.*":    0.10,
  },
  arm_joint_exprs=(r".*shoulder.*", r".*elbow.*", r".*wrist.*"),
  # The soft G1 arm PD gains let the raised push pose sag ~0.28 rad under
  # gravity at zero action, so THEMIS's 0.25 rad would terminate at once.
  arm_max_deviation=0.45,
  # Model mass; whole-body inertia about the CoM in the knees-bent keyframe.
  mass=33.34,
  inertia=(
    (3.523, 0.0,   0.274),
    (0.0,   3.162, 0.0  ),
    (0.274, 0.0,   0.542),
  ),
  com_height=0.76,
  hip_width=0.12,  # sole sites sit at y = ±0.1185 in the default pose
  loco_manip_hip_width=0.12,
  gait_period=0.8,
  # From mjlab's G1 velocity task: the capsule feet need extra CCD
  # iterations and contact-sensor matches.
  ccd_iterations=500,
  contact_sensor_maxmatch=500,
  nconmax=45,
  experiment_name="g1_velocity",
)
