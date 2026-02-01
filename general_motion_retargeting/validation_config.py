"""
Validation configurations for different robots.

Maps robots to their joint limits, keypoint mappings, and collision pairs.
"""

import json
from pathlib import Path

# Joint limits in radians for common humanoid robots
# These are approximate safe ranges; adjust based on actual robot specs

UNITREE_G1_LIMITS = {
    'left_hip_roll': (-0.52, 0.52),
    'left_hip_yaw': (-0.52, 0.52),
    'left_hip_pitch': (-1.57, 0.52),
    'left_knee': (0, 2.79),
    'left_ankle': (-0.52, 0.52),
    'right_hip_roll': (-0.52, 0.52),
    'right_hip_yaw': (-0.52, 0.52),
    'right_hip_pitch': (-1.57, 0.52),
    'right_knee': (0, 2.79),
    'right_ankle': (-0.52, 0.52),
    'left_shoulder_roll': (-2.09, 2.09),
    'left_shoulder_pitch': (-2.09, 2.09),
    'left_shoulder_yaw': (-2.09, 2.09),
    'left_elbow': (-2.09, 2.09),
    'right_shoulder_roll': (-2.09, 2.09),
    'right_shoulder_pitch': (-2.09, 2.09),
    'right_shoulder_yaw': (-2.09, 2.09),
    'right_elbow': (-2.09, 2.09),
}

UNITREE_H1_LIMITS = {
    'left_hip_roll': (-0.87, 0.87),
    'left_hip_yaw': (-0.87, 0.87),
    'left_hip_pitch': (-1.57, 0.87),
    'left_knee': (0, 2.79),
    'left_ankle_pitch': (-0.67, 0.67),
    'left_ankle_roll': (-0.67, 0.67),
    'right_hip_roll': (-0.87, 0.87),
    'right_hip_yaw': (-0.87, 0.87),
    'right_hip_pitch': (-1.57, 0.87),
    'right_knee': (0, 2.79),
    'right_ankle_pitch': (-0.67, 0.67),
    'right_ankle_roll': (-0.67, 0.67),
    'torso': (-0.87, 0.87),
    'left_shoulder_roll': (-2.09, 2.09),
    'left_shoulder_pitch': (-1.57, 1.57),
    'left_elbow': (-2.09, 2.09),
    'right_shoulder_roll': (-2.09, 2.09),
    'right_shoulder_pitch': (-1.57, 1.57),
    'right_elbow': (-2.09, 2.09),
}

BOOSTER_T1_LIMITS = {
    'left_hip_roll': (-1.57, 1.57),
    'left_hip_pitch': (-1.57, 1.57),
    'left_knee': (-2.09, 0),
    'left_ankle_pitch': (-0.87, 0.87),
    'left_ankle_roll': (-0.87, 0.87),
    'right_hip_roll': (-1.57, 1.57),
    'right_hip_pitch': (-1.57, 1.57),
    'right_knee': (-2.09, 0),
    'right_ankle_pitch': (-0.87, 0.87),
    'right_ankle_roll': (-0.87, 0.87),
    'waist_roll': (-0.87, 0.87),
    'waist_pitch': (-0.87, 0.87),
    'waist_yaw': (-0.87, 0.87),
}

BOOSTER_T1_29DOF_INSPIRE_LIMITS = {
    'left_hip_roll': (-1.57, 1.57),
    'left_hip_pitch': (-1.57, 1.57),
    'left_knee': (-2.09, 0),
    'left_ankle_pitch': (-0.87, 0.87),
    'left_ankle_roll': (-0.87, 0.87),
    'right_hip_roll': (-1.57, 1.57),
    'right_hip_pitch': (-1.57, 1.57),
    'right_knee': (-2.09, 0),
    'right_ankle_pitch': (-0.87, 0.87),
    'right_ankle_roll': (-0.87, 0.87),
    'waist_roll': (-0.87, 0.87),
    'waist_pitch': (-0.87, 0.87),
    'waist_yaw': (-0.87, 0.87),
    'left_shoulder_roll': (-2.09, 2.09),
    'left_shoulder_pitch': (-2.09, 2.09),
    'left_shoulder_yaw': (-2.09, 2.09),
    'left_elbow': (-2.09, 2.09),
    'right_shoulder_roll': (-2.09, 2.09),
    'right_shoulder_pitch': (-2.09, 2.09),
    'right_shoulder_yaw': (-2.09, 2.09),
    'right_elbow': (-2.09, 2.09),
}
T1_LIMITS = {
    'waist': (-1.047, 1.047),
    'left_hip_pitch': (-1.8, 1.57),
    'left_hip_roll': (-0.3, 1.57),
    'left_hip_yaw': (-1.0, 1.0),
    'left_knee_pitch': (0.0, 2.34),
    'left_ankle_pitch': (-0.87, 0.35),
    'left_ankle_roll': (-0.44, 0.44),
    'right_hip_pitch': (-1.8, 1.57),
    'right_hip_roll': (-1.57, 0.3),
    'right_hip_yaw': (-1.0, 1.0),
    'right_knee_pitch': (0.0, 2.34),
    'right_ankle_pitch': (-0.87, 0.35),
    'right_ankle_roll': (-0.44, 0.44),
    'head_yaw': (-1.57, 1.57),
    'head_pitch': (-0.35, 1.22),
    'left_shoulder_pitch': (-3.3469, 1.2255),
    'left_shoulder_roll': (-1.7239, 1.7357),
    'left_elbow_pitch': (-2.3249, 2.2581),
    'left_elbow_yaw': (-2.1418, 1.6978),
    'left_wrist_pitch': (-2.6164, 2.6209),
    'left_wrist_yaw': (-1.861, 1.4815),
    'left_hand_roll': (-1.0348, 1.6066),
    'right_shoulder_pitch': (-3.3953, 1.1954),
    'right_shoulder_roll': (-1.619, 1.7392),
    'right_elbow_pitch': (-2.3154, 2.2749),
    'right_elbow_yaw': (-1.6917, 2.1578),
    'right_wrist_pitch': (-2.6133, 2.627),
    'right_wrist_yaw': (-1.4502, 1.8935),
    'right_hand_roll': (-1.6768, 0.9993),
}


# Keypoint mappings: which body corresponds to each tracking point
KEYPOINT_MAPS = {
    'unitree_g1': {
        'pelvis': 'pelvis',
        'head': 'head',
        'left_hand': 'left_hand',
        'right_hand': 'right_hand',
        'left_foot': 'left_foot',
        'right_foot': 'right_foot',
    },
    'unitree_h1': {
        'pelvis': 'pelvis',
        'head': 'head',
        'left_hand': 'left_hand',
        'right_hand': 'right_hand',
        'left_foot': 'left_foot',
        'right_foot': 'right_foot',
    },
    'booster_t1': {
        'pelvis': 'Waist',
        'head': 'Head',
        'left_hand': 'LeftHand',
        'right_hand': 'RightHand',
        'left_foot': 'LeftFoot',
        'right_foot': 'RightFoot',
    },
    'booster_t1_29dof_inspire_custom': {
        'pelvis': 'Waist',
        'head': 'Head',
        'left_hand': 'LeftHand',
        'right_hand': 'RightHand',
        'left_foot': 'LeftFoot',
        'right_foot': 'RightFoot',
    },
    't1': {
        'pelvis': 'Waist',
        'head': 'Head',
        'left_hand': 'LeftHand',
        'right_hand': 'RightHand',
        'left_foot': 'LeftFoot',
        'right_foot': 'RightFoot',
    },
}

# Self-collision pairs to check: (body1_name, body2_name)
COLLISION_PAIRS = {
    'unitree_g1': [
        ('left_foot', 'right_foot'),
        ('left_hand', 'torso'),
        ('right_hand', 'torso'),
        ('left_thigh', 'torso'),
        ('right_thigh', 'torso'),
    ],
    'unitree_h1': [
        ('left_foot', 'right_foot'),
        ('left_hand', 'torso'),
        ('right_hand', 'torso'),
        ('left_thigh', 'torso'),
        ('right_thigh', 'torso'),
    ],
    'booster_t1': [
        ('LeftFoot', 'RightFoot'),
        ('LeftHand', 'Waist'),
        ('RightHand', 'Waist'),
        ('LeftThigh', 'Waist'),
        ('RightThigh', 'Waist'),
    ],
    'booster_t1_29dof_inspire_custom': [
        ('LeftFoot', 'RightFoot'),
        ('LeftHand', 'Waist'),
        ('RightHand', 'Waist'),
        ('LeftThigh', 'Waist'),
        ('RightThigh', 'Waist'),
    ],
    't1': [
        ('left_foot_link', 'right_foot_link'),
        ('left_inspire_hand', 'Trunk'),
        ('right_inspire_hand', 'Trunk'),
    ],
}

# ============================================================================
# COLLISION GROUPS (Kheiron-style group-based collision detection)
# ============================================================================
# Define body groups and which group pairs should be checked for collisions.
# This is more targeted than checking all body pairs.

COLLISION_GROUPS = {
    'booster_t1_29dof_inspire_custom': {
        # Body name -> Group name mapping
        'body_to_group': {
            # Head
            'H1': 'head',
            'H2': 'head',
            # Torso
            'Trunk': 'torso',
            'Waist': 'torso',
            # Upper Arm Left (first cylinder out of torso)
            'AL1': 'upper_arm_L',
            # Upper Arm Right
            'AR1': 'upper_arm_R',
            # Forearm Left (includes hand and fingers)
            'AL2': 'forearm_L',
            'AL3': 'forearm_L',
            'AL4': 'forearm_L',
            'AL5': 'forearm_L',
            'AL6': 'forearm_L',
            'left_hand_link': 'forearm_L',
            'left_thumb_1': 'forearm_L',
            'left_thumb_2': 'forearm_L',
            'left_thumb_3': 'forearm_L',
            'left_thumb_4': 'forearm_L',
            'left_index_1': 'forearm_L',
            'left_index_2': 'forearm_L',
            'left_middle_1': 'forearm_L',
            'left_middle_2': 'forearm_L',
            'left_ring_1': 'forearm_L',
            'left_ring_2': 'forearm_L',
            'left_little_1': 'forearm_L',
            'left_little_2': 'forearm_L',
            # Forearm Right (includes hand and fingers)
            'AR2': 'forearm_R',
            'AR3': 'forearm_R',
            'AR4': 'forearm_R',
            'AR5': 'forearm_R',
            'AR6': 'forearm_R',
            'right_hand_link': 'forearm_R',
            'right_thumb_1': 'forearm_R',
            'right_thumb_2': 'forearm_R',
            'right_thumb_3': 'forearm_R',
            'right_thumb_4': 'forearm_R',
            'right_index_1': 'forearm_R',
            'right_index_2': 'forearm_R',
            'right_middle_1': 'forearm_R',
            'right_middle_2': 'forearm_R',
            'right_ring_1': 'forearm_R',
            'right_ring_2': 'forearm_R',
            'right_little_1': 'forearm_R',
            'right_little_2': 'forearm_R',
            # Leg Left
            'Hip_Pitch_Left': 'leg_L',
            'Hip_Roll_Left': 'leg_L',
            'Hip_Yaw_Left': 'leg_L',
            'Shank_Left': 'leg_L',
            'Ankle_Cross_Left': 'leg_L',
            'left_foot_link': 'leg_L',
            # Leg Right
            'Hip_Pitch_Right': 'leg_R',
            'Hip_Roll_Right': 'leg_R',
            'Hip_Yaw_Right': 'leg_R',
            'Shank_Right': 'leg_R',
            'Ankle_Cross_Right': 'leg_R',
            'right_foot_link': 'leg_R',
        },
        # Group pairs to check for collisions
        # Format: (group1, group2)
        'group_pairs': [
            # Forearm vs opposite upper arm (cross-body)
            ('forearm_L', 'upper_arm_R'),
            ('forearm_R', 'upper_arm_L'),
            # Forearm vs forearm (both arms)
            ('forearm_L', 'forearm_R'),
            # Forearm vs torso
            ('forearm_L', 'torso'),
            ('forearm_R', 'torso'),
            # Forearm vs head
            ('forearm_L', 'head'),
            ('forearm_R', 'head'),
            # Forearm vs legs
            ('forearm_L', 'leg_L'),
            ('forearm_L', 'leg_R'),
            ('forearm_R', 'leg_L'),
            ('forearm_R', 'leg_R'),
            # Leg vs leg (feet collision)
            ('leg_L', 'leg_R'),
        ],
        # Bodies to skip when checking upper_arm vs torso
        # (to avoid false positives at shoulder joint)
        'skip_pairs': [
            ('AL1', 'Trunk'),  # Upper arm attached to trunk
            ('AR1', 'Trunk'),
        ],
    },
    't1': {
        'body_to_group': {
            # Head
            'H1': 'head',
            'H2': 'head',
            # Torso
            'Trunk': 'torso',
            'Waist': 'torso',
            # Upper Arm Left
            'AL1': 'upper_arm_L',
            # Upper Arm Right
            'AR1': 'upper_arm_R',
            # Forearm Left (includes hand and fingers)
            'AL2': 'forearm_L', 'AL3': 'forearm_L', 'AL4': 'forearm_L',
            'AL5': 'forearm_L', 'AL6': 'forearm_L',
            'left_inspire_hand': 'forearm_L',
            'left_thumb_1': 'forearm_L', 'left_thumb_2': 'forearm_L',
            'left_thumb_3': 'forearm_L', 'left_thumb_4': 'forearm_L',
            'left_index_1': 'forearm_L', 'left_index_2': 'forearm_L',
            'left_middle_1': 'forearm_L', 'left_middle_2': 'forearm_L',
            'left_ring_1': 'forearm_L', 'left_ring_2': 'forearm_L',
            'left_little_1': 'forearm_L', 'left_little_2': 'forearm_L',
            # Forearm Right (includes hand and fingers)
            'AR2': 'forearm_R', 'AR3': 'forearm_R', 'AR4': 'forearm_R',
            'AR5': 'forearm_R', 'AR6': 'forearm_R',
            'right_inspire_hand': 'forearm_R',
            'right_thumb_1': 'forearm_R', 'right_thumb_2': 'forearm_R',
            'right_thumb_3': 'forearm_R', 'right_thumb_4': 'forearm_R',
            'right_index_1': 'forearm_R', 'right_index_2': 'forearm_R',
            'right_middle_1': 'forearm_R', 'right_middle_2': 'forearm_R',
            'right_ring_1': 'forearm_R', 'right_ring_2': 'forearm_R',
            'right_little_1': 'forearm_R', 'right_little_2': 'forearm_R',
            # Leg Left (hip, thigh, shank, ankle, foot)
            'Hip_Pitch_Left': 'leg_L',
            'Hip_Roll_Left': 'leg_L',
            'Hip_Yaw_Left': 'leg_L',
            'Shank_Left': 'leg_L',
            'Ankle_Cross_Left': 'leg_L',
            'left_foot_link': 'leg_L',
            # Leg Right (hip, thigh, shank, ankle, foot)
            'Hip_Pitch_Right': 'leg_R',
            'Hip_Roll_Right': 'leg_R',
            'Hip_Yaw_Right': 'leg_R',
            'Shank_Right': 'leg_R',
            'Ankle_Cross_Right': 'leg_R',
            'right_foot_link': 'leg_R',
        },
        'group_pairs': [
            # Forearm vs opposite upper arm (cross-body)
            ('forearm_L', 'upper_arm_R'), ('forearm_R', 'upper_arm_L'),
            # Forearm vs forearm (both arms)
            ('forearm_L', 'forearm_R'),
            # Forearm vs torso
            ('forearm_L', 'torso'), ('forearm_R', 'torso'),
            # Forearm vs head
            ('forearm_L', 'head'), ('forearm_R', 'head'),
            # Forearm vs legs
            ('forearm_L', 'leg_L'), ('forearm_L', 'leg_R'),
            ('forearm_R', 'leg_L'), ('forearm_R', 'leg_R'),
            # Leg vs leg (feet collision)
            ('leg_L', 'leg_R'),
        ],
        # Bodies to skip when checking upper_arm vs torso
        # (to avoid false positives at shoulder joint)
        'skip_pairs': [
            ('AL1', 'Trunk'),  # Upper arm attached to trunk
            ('AR1', 'Trunk'),
            ('Hip_Pitch_Left', 'Waist'),  # Hip attached to waist
            ('Hip_Pitch_Right', 'Waist'),
        ],
    },
}

# Foot body names for each robot
FOOT_NAMES = {
    'unitree_g1': ['left_foot', 'right_foot'],
    'unitree_h1': ['left_foot', 'right_foot'],
    'booster_t1': ['LeftFoot', 'RightFoot'],
    'booster_t1_29dof_inspire_custom': ['left_foot_link', 'right_foot_link'],
    't1': ['left_foot_link', 'right_foot_link'],
}

# Robot XML file paths and joint names
ROBOT_CONFIGS = {
    'unitree_g1': {
        'xml': 'assets/unitree_g1/g1_mocap_29dof.xml',
        'joint_limits': UNITREE_G1_LIMITS,
        'keypoint_map': KEYPOINT_MAPS['unitree_g1'],
        'collision_pairs': COLLISION_PAIRS['unitree_g1'],
        'collision_groups': None,  # No group config yet
        'foot_names': FOOT_NAMES['unitree_g1'],
    },
    'unitree_h1': {
        'xml': 'assets/unitree_h1/h1.xml',
        'joint_limits': UNITREE_H1_LIMITS,
        'keypoint_map': KEYPOINT_MAPS['unitree_h1'],
        'collision_pairs': COLLISION_PAIRS['unitree_h1'],
        'collision_groups': None,  # No group config yet
        'foot_names': FOOT_NAMES['unitree_h1'],
    },
    'booster_t1': {
        'xml': 'assets/booster_t1/T1_serial.xml',
        'joint_limits': BOOSTER_T1_LIMITS,
        'keypoint_map': KEYPOINT_MAPS['booster_t1'],
        'collision_pairs': COLLISION_PAIRS['booster_t1'],
        'collision_groups': None,  # No group config yet
        'foot_names': FOOT_NAMES['booster_t1'],
    },
    'booster_t1_29dof_inspire_custom': {
        'xml': 'assets/booster_t1_29dof_inspire_custom/robot.xml',
        'joint_limits': BOOSTER_T1_29DOF_INSPIRE_LIMITS,
        'keypoint_map': KEYPOINT_MAPS['booster_t1_29dof_inspire_custom'],
        'collision_pairs': COLLISION_PAIRS['booster_t1_29dof_inspire_custom'],
        'collision_groups': COLLISION_GROUPS.get('booster_t1_29dof_inspire_custom'),
        'foot_names': FOOT_NAMES['booster_t1_29dof_inspire_custom'],
    },
    't1': {
        'xml': 'assets/t1/t1_robot.xml',
        'joint_limits': T1_LIMITS,
        'keypoint_map': KEYPOINT_MAPS['t1'],
        'collision_pairs': COLLISION_PAIRS['t1'],
        'collision_groups': COLLISION_GROUPS.get('t1'),
        'foot_names': FOOT_NAMES['t1'],
    }
}


def get_robot_config(robot_name: str) -> dict:
    """Get configuration for a specific robot."""
    if robot_name not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot: {robot_name}. Available: {list(ROBOT_CONFIGS.keys())}")
    return ROBOT_CONFIGS[robot_name]
