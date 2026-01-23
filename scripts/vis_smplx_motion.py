import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to visualize.",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--save_video",
        default=None,
        help="Path to save video visualization.",
    )
    
    args = parser.parse_args()
    
    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    print(f"Loading SMPL-X data from {args.smplx_file}...")
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    print(f"Human height: {actual_human_height:.3f} m")
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    
    print(f"Number of frames: {len(smplx_data_frames)}")
    print(f"FPS: {aligned_fps}")
    
    # Extract joint positions from first frame to visualize skeleton structure
    first_frame = smplx_data_frames[0]
    print("\nSkeleton joints:")
    print(f"  Frame structure: {type(first_frame)}")
    if isinstance(first_frame, dict):
        for joint_name, joint_data in first_frame.items():
            if isinstance(joint_data, dict) and 'pos' in joint_data:
                pos = joint_data['pos']
                print(f"  {joint_name}: position {pos}")
            else:
                print(f"  {joint_name}: {joint_data}")
    
    # Visualize using matplotlib
    if args.save_video is None:
        print("\nVisualization complete! Use --save_video to record a video.")
    else:
        try:
            import cv2
            print(f"Recording video to {args.save_video}...")
            
            # Get video parameters
            fps = int(aligned_fps)
            height, width = 800, 800
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))
            
            # Create figure
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Key joints to visualize (skeleton structure)
            key_joints = [
                'pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'head',
                'left_shoulder', 'left_elbow', 'left_wrist',
                'right_shoulder', 'right_elbow', 'right_wrist',
                'left_hip', 'left_knee', 'left_ankle',
                'right_hip', 'right_knee', 'right_ankle'
            ]
            
            # Connections between joints (skeleton edges)
            skeleton_edges = [
                ('pelvis', 'spine1'),
                ('spine1', 'spine2'),
                ('spine2', 'spine3'),
                ('spine3', 'neck'),
                ('neck', 'head'),
                ('spine3', 'left_shoulder'),
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('spine3', 'right_shoulder'),
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
                ('pelvis', 'left_hip'),
                ('left_hip', 'left_knee'),
                ('left_knee', 'left_ankle'),
                ('pelvis', 'right_hip'),
                ('right_hip', 'right_knee'),
                ('right_knee', 'right_ankle'),
            ]
            
            for frame_idx, frame in enumerate(smplx_data_frames):
                ax.clear()
                
                # Extract positions
                positions = {}
                for joint_name in key_joints:
                    if joint_name in frame:
                        joint_data = frame[joint_name]
                        # joint_data is a tuple of (pos_array, rot_array)
                        if isinstance(joint_data, tuple) and len(joint_data) >= 1:
                            positions[joint_name] = joint_data[0]
                        elif isinstance(joint_data, np.ndarray):
                            positions[joint_name] = joint_data
                
                # Plot joints
                for joint_name, pos in positions.items():
                    ax.scatter(*pos, s=50, c='red')
                    ax.text(pos[0], pos[1], pos[2], joint_name, fontsize=8)
                
                # Plot skeleton edges
                for joint1, joint2 in skeleton_edges:
                    if joint1 in positions and joint2 in positions:
                        pos1 = positions[joint1]
                        pos2 = positions[joint2]
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 'b-', linewidth=2)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'SMPL-X Frame {frame_idx+1}/{len(smplx_data_frames)}')
                
                # Set consistent axis limits
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([0, 2])
                
                plt.tight_layout()
                
                # Save frame to video using matplotlib backend
                fig.savefig('/tmp/frame.png', dpi=100, bbox_inches='tight')
                frame_img = cv2.imread('/tmp/frame.png')
                frame_img = cv2.resize(frame_img, (width, height))
                out.write(frame_img)
                
                if (frame_idx + 1) % 30 == 0:
                    print(f"  Processed {frame_idx+1}/{len(smplx_data_frames)} frames")
            
            out.release()
            plt.close(fig)
            print(f"Video saved to {args.save_video}")
            
        except ImportError:
            print("OpenCV not installed. Cannot save video. Install with: pip install opencv-python")
