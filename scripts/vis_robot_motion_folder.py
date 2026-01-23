#!/usr/bin/env python3
import argparse
import pathlib
import os
from pathlib import Path
import numpy as np
from general_motion_retargeting import RobotMotionViewer
import pickle
from tqdm import tqdm

def visualize_motion_folder(robot_type, motion_folder, output_video_dir=None, record_video=False):
    """Visualize all robot motions in a folder"""
    
    motion_folder = Path(motion_folder)
    pkl_files = sorted(motion_folder.glob("*.pkl"))
    
    if not pkl_files:
        print(f"No .pkl files found in {motion_folder}")
        return
    
    print(f"Found {len(pkl_files)} motion files to visualize")
    
    for motion_file in tqdm(pkl_files, desc="Visualizing motions"):
        print(f"\n{'='*80}")
        print(f"Processing: {motion_file.name}")
        print(f"{'='*80}")
        
        # Load motion data
        with open(motion_file, "rb") as f:
            motion_data = pickle.load(f)
        
        motion_fps = motion_data["fps"]
        root_pos = motion_data["root_pos"]
        root_rot = motion_data["root_rot"]
        dof_pos = motion_data["dof_pos"]
        
        # Create viewer
        video_path = None
        if record_video and output_video_dir:
            os.makedirs(output_video_dir, exist_ok=True)
            video_name = motion_file.stem + ".mp4"
            video_path = os.path.join(output_video_dir, video_name)
        
        robot_viewer = RobotMotionViewer(
            robot_type=robot_type,
            motion_fps=motion_fps,
            transparent_robot=0,
            record_video=record_video,
            video_path=video_path,
        )
        
        # Play motion
        frame_count = 0
        max_frames = len(root_pos)
        
        try:
            while frame_count < max_frames:
                robot_viewer.step(
                    root_pos=root_pos[frame_count],
                    root_rot=root_rot[frame_count],
                    dof_pos=dof_pos[frame_count],
                )
                frame_count += 1
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            robot_viewer.close()
            if record_video and video_path:
                print(f"Saved video to: {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize all robot motions in a folder")
    
    parser.add_argument(
        "--robot",
        default="booster_t1_29dof_inspire_custom",
        help="Robot type"
    )
    
    parser.add_argument(
        "--motion_folder",
        required=True,
        help="Path to folder containing .pkl motion files"
    )
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
        help="Record videos"
    )
    
    parser.add_argument(
        "--output_video_dir",
        default="videos/retargeted_motions",
        help="Directory to save output videos"
    )
    
    args = parser.parse_args()
    
    visualize_motion_folder(
        robot_type=args.robot,
        motion_folder=args.motion_folder,
        output_video_dir=args.output_video_dir,
        record_video=args.record_video
    )
