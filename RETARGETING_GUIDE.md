# GMR Motion Retargeting: SMPL-X ACCAD to Booster T1 Inspire Custom

## Installation

```bash
git clone https://github.com/AnujithM/GMR-T1-G1.git
cd GMR-T1-G1
conda create -n gmr python=3.10
conda activate gmr
pip install -e .
```

## Setup Data

1. **Download SMPL-X models** (register at https://smpl-x.is.tue.mpg.de/):
```bash
mkdir -p assets/body_models/smplx
# Place SMPLX_NEUTRAL.pkl here
```

2. **Download AMASS ACCAD data** (https://amass.is.tue.mpg.de/):
```bash
mkdir -p ~/datasets/AMASS/ACCAD
# Extract ACCAD subsets (Male1General_c3d, Male1Walking_c3d, etc.)
```

## Retarget Single Motion

```bash
python scripts/smplx_to_robot.py \
    --smplx_file ~/datasets/AMASS/ACCAD/Male1Walking_c3d/Walk_B15_-_Walk_turn_around_stageii.npz \
    --robot t1 \
    --save_path outputs/retargeted_motion
```

Output: `outputs/retargeted_motion/Walk_B15_-_Walk_turn_around_stageii.pkl`

## Batch Retarget Folder

```bash
python scripts/smplx_to_robot_dataset.py \
    --dataset_path ~/datasets/AMASS/ACCAD/Male1General_c3d \
    --robot t1 \
    --save_path outputs/retargeted_male1general_t1
```

## Visualize Results

**SMPL-X visualization:**
```bash
python scripts/visualize.py
```

**Retargeted robot motion:**
```bash
python scripts/vis_robot_motion.py \
    --robot_motion_path outputs/retargeted_motion/Walk_B15_-_Walk_turn_around_stageii.pkl \
    --robot t1 \
    --record_video \
    --video_path videos/robot_motion.mp4
```

## Output Format

Retargeted motions are saved as `.pkl` files containing:
- `fps`: 30
- `root_pos`: Root position (N, 3)
- `root_rot`: Root rotation quaternion xyzw (N, 4)
- `dof_pos`: Joint DOF positions (N, DOF)


## IK Config

The retargeting uses: `general_motion_retargeting/ik_configs/smplx_to_t1_inspire.json`

Maps SMPL-X (55 joints) → Booster T1 Inspire Custom (59 DOF):
- Waist: 6 DOF
- Left/Right Leg: 6 DOF each
- Left/Right arm: 9 DOF each
- Head: 2 DOF
- Inspire Hands: 20 DOF total
