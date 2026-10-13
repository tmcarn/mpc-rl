"""Unitree G1 29-DOF humanoid constants for the MPC-guided tasks.

Reuses mjlab's G1 model, motor specs and actuator configs
(``mjlab.asset_zoo.robots.unitree_g1``) and adds the extra MJCF elements the
THEMIS tasks expect, under the same names so the shared env builders work
unchanged:

* ``hand_contact_L`` / ``hand_contact_R`` — sphere geoms on the wrist-yaw
  links used for hand-box contact sensing (loco-manipulation).
* ``left_hand`` / ``right_hand`` — sites at the same point, used by the
  loco-manipulation MPC as the hand force application points.
* ``chest_camera`` — site on the torso used by the hand-slip penalty.
"""

import mujoco
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as _g1
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Unitree's own MJCF has no hand contact geoms; the palm sites sit 0.08 m
# out along the wrist-yaw link x-axis, at the rubber hand.
_PALM_POS = (0.08, 0.0, 0.0)
_HAND_CONTACT_RADIUS = 0.04


def get_spec() -> mujoco.MjSpec:
  spec = _g1.get_spec()
  for side, suffix in (("left", "L"), ("right", "R")):
    body = spec.body(f"{side}_wrist_yaw_link")
    body.add_geom(
      name=f"hand_contact_{suffix}",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(_HAND_CONTACT_RADIUS, 0, 0),
      pos=_PALM_POS,
      mass=0.0,
      density=0.0,
      rgba=(0.9, 0.3, 0.1, 0.5),
      group=3,
    )
    body.add_site(name=f"{side}_hand", pos=_PALM_POS, size=(0.015, 0, 0), group=5)
  # Front of the chest, pitched down ~40 deg like the THEMIS chest camera.
  spec.body("torso_link").add_site(
    name="chest_camera",
    pos=(0.08, 0.0, 0.30),
    zaxis=(0.6427876097, 0.0, -0.7660444431),
    size=(0.015, 0, 0),
    group=5,
  )
  return spec


##
# Joint ordering — Unitree SDK motor order:
#   left leg → right leg → waist → left arm → right arm
##
JOINT_NAMES_EXPR = [
  # Left leg (6)
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  # Right leg (6)
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  # Waist (3)
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  # Left arm (7)
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  # Right arm (7)
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]


##
# Keyframes.
##

# Walking default pose — mjlab's knees-bent G1 keyframe (elbows bent).
HOME_KEYFRAME_WALKING = _g1.KNEES_BENT_KEYFRAME

# Loco-manipulation default pose — arms raised forward so the palms sit
# ~0.3 m in front of the pelvis at ~0.85 m height, ready to push.
HOME_KEYFRAME_LOCO_MANIP = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_shoulder_pitch_joint": -0.3,
    "left_shoulder_roll_joint": 0.15,
    "right_shoulder_roll_joint": -0.15,
    ".*_elbow_joint": 0.3,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

# mjlab's G1 default: all collision geoms enabled including self collisions,
# condim=3 on the 7 foot capsules per side and condim=1 elsewhere.
FULL_COLLISION = _g1.FULL_COLLISION

# Hand contact spheres: frictional (condim=3) so hand-box push forces have a
# tangential component.  Collide with the box / world only (contype=0 keeps
# them out of robot self-collision with the arm links they overlap).
HAND_CONTACT_COLLISION = CollisionCfg(
  geom_names_expr=("hand_contact_.*",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(1.0, 0.005, 0.0001),
  disable_other_geoms=False,
)


##
# Final config.
##
G1_ARTICULATION = _g1.G1_ARTICULATION
G1_ACTION_SCALE = _g1.G1_ACTION_SCALE


def get_g1_robot_cfg(loco_manip: bool = False) -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Args:
    loco_manip: When ``True``, initialise with the loco-manipulation default
      pose (arms raised forward for pushing); otherwise use the walking pose.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME_LOCO_MANIP if loco_manip else HOME_KEYFRAME_WALKING,
    collisions=(FULL_COLLISION, HAND_CONTACT_COLLISION),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg(loco_manip=True))
  viewer.launch(robot.spec.compile())
