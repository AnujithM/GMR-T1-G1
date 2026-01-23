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
    --robot booster_t1_29dof_inspire_custom \
    --output_dir outputs/retargeted_motion
```

Output: `outputs/retargeted_motion/Walk_B15_-_Walk_turn_around_stageii.pkl`

## Batch Retarget Folder

```bash
python scripts/smplx_to_robot_dataset.py \
    --dataset_path ~/datasets/AMASS/ACCAD/Male1General_c3d \
    --robot booster_t1_29dof_inspire_custom \
    --output_dir outputs/retargeted_male1general_t1_inspire
```

## Visualize Results

**SMPL-X skeleton visualization:**
```bash
python scripts/vis_smplx_motion.py \
    --smplx_file ~/datasets/AMASS/ACCAD/Male1Walking_c3d/Walk_B15_-_Walk_turn_around_stageii.npz \
    --save_video videos/skeleton.mp4
```

**Retargeted robot motion:**
```bash
python scripts/vis_robot_motion.py \
    --motion_file outputs/retargeted_motion/Walk_B15_-_Walk_turn_around_stageii.pkl \
    --robot booster_t1_29dof_inspire_custom \
    --record_video videos/robot_motion.mp4
```

## Output Format

Retargeted motions are saved as `.pkl` files containing:
- `fps`: 30
- `root_pos`: Root position (N, 3)
- `root_rot`: Root rotation quaternion xyzw (N, 4)
- `dof_pos`: Joint angles for 29 DOF (N, 29)


## IK Config

The retargeting uses: `general_motion_retargeting/ik_configs/smplx_to_booster_t1_29dof_inspire_custom.json`

Maps SMPL-X (55 joints) → Booster T1 Inspire Custom (29 DOF):
- Waist: 6 DOF
- Left/Right Leg: 7 DOF each
- Head: 2 DOF
- Inspire Hands: 40 DOF total (20 per hand)
