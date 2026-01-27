#!/usr/bin/env python3
"""
Example script for validating GMR retargeted motions.

Usage:
    python validate_motion.py --motion motion.pkl --robot unitree_g1 --output results/
    python validate_motion.py --motion motion.pkl --robot unitree_g1 --output results/ --show-plots
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "general_motion_retargeting"))

from motion_validator import MotionValidator
from validation_config import get_robot_config, ROBOT_CONFIGS


def load_gmr_motion(motion_file: str):
    """
    Load motion data from GMR pickle file.
    
    Returns:
        dict: motion_data
        float: fps
        np.ndarray: root_pose (T, 7) or None
        dict: foot_poses mapping foot names to (T, 7) arrays, or None
        np.ndarray: joint_positions (T, n_joints)
    """
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
    
    fps = motion_data.get("fps", 30.0)
    
    # Extract joint positions
    joint_positions = motion_data.get("dof_pos", None)
    if joint_positions is None:
        raise ValueError("Motion data must contain 'dof_pos' field")
    
    # Extract root pose if available
    root_pose = None
    if "root_pos" in motion_data and "root_rot" in motion_data:
        root_pos = motion_data["root_pos"]  # (T, 3)
        root_rot = motion_data["root_rot"]  # (T, 4) in xyzw format
        
        # Convert to wxyz format if needed
        if root_rot.shape[1] == 4:
            root_rot_wxyz = np.column_stack([
                root_rot[:, 3],  # w
                root_rot[:, 0],  # x
                root_rot[:, 1],  # y
                root_rot[:, 2],  # z
            ])
        else:
            root_rot_wxyz = root_rot
        
        root_pose = np.column_stack([root_pos, root_rot_wxyz[:, 1:], root_rot_wxyz[:, 0]])
        root_pose = root_pose[:, [0, 1, 2, 3, 4, 5, 6]]  # [x, y, z, qx, qy, qz, qw]
    
    # Extract foot poses if available
    foot_poses = None
    if "foot_poses" in motion_data:
        foot_poses = motion_data["foot_poses"]
    # Note: local_body_pos contains LOCAL positions, not global.
    # Global foot positions will be computed via MuJoCo FK if foot_poses is None
    
    return motion_data, fps, root_pose, foot_poses, joint_positions


def load_target_keypoints(keypoint_file: str, fps: float = 30.0):
    """
    Load target keypoints from file (CSV or NPZ format).
    
    Expected format:
        CSV: frame, keypoint_name, x, y, z
        NPZ: dict with keypoint names as keys, arrays of shape (T, 3)
    
    Returns:
        dict: mapping keypoint names to (T, 3) position arrays
    """
    keypoint_file = Path(keypoint_file)
    
    if keypoint_file.suffix == '.npz':
        data = np.load(keypoint_file)
        return {name: data[name] for name in data.files}
    elif keypoint_file.suffix == '.csv':
        # Parse CSV with columns: frame, keypoint_name, x, y, z
        import csv
        keypoints = {}
        with open(keypoint_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['keypoint_name']
                if name not in keypoints:
                    keypoints[name] = []
                keypoints[name].append([
                    float(row['x']),
                    float(row['y']),
                    float(row['z']),
                ])
        
        # Convert to arrays
        return {name: np.array(positions) for name, positions in keypoints.items()}
    else:
        raise ValueError(f"Unsupported file format: {keypoint_file.suffix}")


def validate_motion(
    motion_file: str,
    robot: str,
    keypoint_file: str = None,
    output_dir: str = "validation_output",
    verbose: bool = True,
    save_plots: bool = False,
):
    """
    Validate a single motion file.
    
    Args:
        motion_file: Path to GMR motion pickle file
        robot: Robot name (e.g., 'unitree_g1')
        keypoint_file: Optional path to target keypoints
        output_dir: Directory to save reports
        verbose: Print detailed output
        save_plots: Generate and save diagnostic plots
    
    Returns:
        ValidationResult: Complete validation results
    """
    # Load robot config
    config = get_robot_config(robot)
    xml_path = Path(__file__).parent.parent / config['xml']
    
    if not xml_path.exists():
        raise FileNotFoundError(f"Robot XML not found: {xml_path}")
    
    # Load motion data
    if verbose:
        print(f"\n{'='*70}")
        print(f"Loading motion from: {motion_file}")
    
    motion_data, fps, root_pose, foot_poses, joint_positions = load_gmr_motion(motion_file)
    
    if verbose:
        print(f"Loaded motion:")
        print(f"  - Frames: {joint_positions.shape[0]}")
        print(f"  - Joints: {joint_positions.shape[1]}")
        print(f"  - FPS: {fps}")
        if root_pose is not None:
            print(f"  - Root pose: {root_pose.shape}")
    
    # Load target keypoints if provided
    target_keypoints = None
    if keypoint_file:
        if verbose:
            print(f"Loading keypoints from: {keypoint_file}")
        target_keypoints = load_target_keypoints(keypoint_file, fps)
        if verbose:
            print(f"  - Keypoints: {list(target_keypoints.keys())}")
    
    # Create validator
    validator = MotionValidator(
        robot_xml=str(xml_path),
        keypoint_map=config['keypoint_map'],
        collision_pairs=config['collision_pairs'],
        collision_groups=config.get('collision_groups'),
        foot_names=config['foot_names'],
        ground_z=0.0,
        verbose=verbose,
    )
    
    # Extract joint limits from MuJoCo model
    joint_limits = validator.get_joint_limits_from_model()
    if not joint_limits:
        raise RuntimeError(f"Failed to extract joint limits from MuJoCo model: {xml_path}")
    
    # Run validation
    if verbose:
        print(f"\nRunning validation checks...")
    
    result = validator.validate(
        motion_file=motion_file,
        joint_positions=joint_positions,
        root_pose=root_pose,
        foot_poses=foot_poses,
        target_keypoints=target_keypoints,
        fps=fps,
        joint_limits=joint_limits,
    )
    
    # Print summary
    print(result.summary)
    
    # Save reports
    validator.save_reports(result, output_dir)
    
    # Generate plots
    if save_plots:
        try:
            validator.plot_motion(result, joint_positions, output_dir)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    return result


def batch_validate(
    motion_dir: str,
    robot: str,
    output_dir: str = "validation_output",
    verbose: bool = True,
):
    """
    Validate all motion files in a directory.
    
    Args:
        motion_dir: Directory containing .pkl motion files
        robot: Robot name
        output_dir: Directory to save reports
        verbose: Print detailed output
    
    Returns:
        list: ValidationResult objects for all motions
    """
    motion_dir = Path(motion_dir)
    results = []
    
    pkl_files = sorted(motion_dir.glob("*.pkl"))
    
    if not pkl_files:
        print(f"No .pkl files found in {motion_dir}")
        return results
    
    print(f"\nFound {len(pkl_files)} motion files to validate")
    
    for i, pkl_file in enumerate(pkl_files, 1):
        print(f"\n[{i}/{len(pkl_files)}] Validating {pkl_file.name}...")
        try:
            result = validate_motion(
                motion_file=str(pkl_file),
                robot=robot,
                output_dir=output_dir,
                verbose=verbose,
                save_plots=False,
            )
            results.append(result)
        except Exception as e:
            print(f"Error validating {pkl_file.name}: {e}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("Summary of all validations:")
    print(f"{'='*70}")
    print(f"{'Clip Name':<30} {'Duration':>10} {'Status':>15}")
    print("-" * 70)
    
    passed = 0
    for result in results:
        status = "✓ PASS" if result.all_passed else "✗ FAIL"
        if result.all_passed:
            passed += 1
        print(f"{result.clip_name:<30} {result.duration_sec:>9.2f}s {status:>15}")
    
    print("-" * 70)
    print(f"Passed: {passed}/{len(results)}")
    print(f"{'='*70}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate GMR retargeted motion files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file:
    python validate_motion.py --motion motion.pkl --robot unitree_g1
  
  Batch validation:
    python validate_motion.py --motion-dir outputs/ --robot unitree_g1
  
  With keypoint tracking:
    python validate_motion.py --motion motion.pkl --robot unitree_g1 --keypoints keypoints.npz
  
  Save plots:
    python validate_motion.py --motion motion.pkl --robot unitree_g1 --plots
        """)
    
    parser.add_argument('--motion', type=str, help='Path to single .pkl motion file')
    parser.add_argument('--motion-dir', type=str, help='Directory with .pkl motion files for batch validation')
    parser.add_argument('--robot', type=str, required=True, 
                       choices=list(ROBOT_CONFIGS.keys()),
                       help='Robot model')
    parser.add_argument('--keypoints', type=str, help='Path to target keypoints (CSV or NPZ)')
    parser.add_argument('--output', type=str, default='validation_output',
                       help='Output directory for reports (default: validation_output)')
    parser.add_argument('--plots', action='store_true', help='Generate diagnostic plots')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.motion and not args.motion_dir:
        parser.error("Must specify either --motion or --motion-dir")
    
    if args.motion and args.motion_dir:
        parser.error("Cannot specify both --motion and --motion-dir")
    
    # Run validation
    try:
        if args.motion:
            # Single file validation
            if not Path(args.motion).exists():
                print(f"Error: Motion file not found: {args.motion}")
                sys.exit(1)
            
            validate_motion(
                motion_file=args.motion,
                robot=args.robot,
                keypoint_file=args.keypoints,
                output_dir=args.output,
                verbose=not args.quiet,
                save_plots=args.plots,
            )
        else:
            # Batch validation
            if not Path(args.motion_dir).exists():
                print(f"Error: Directory not found: {args.motion_dir}")
                sys.exit(1)
            
            batch_validate(
                motion_dir=args.motion_dir,
                robot=args.robot,
                output_dir=args.output,
                verbose=not args.quiet,
            )
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
