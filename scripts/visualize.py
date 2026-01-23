import time
import numpy as np
import torch
import trimesh
import pyrender
from smplx import SMPL
import open3d as o3d
import torch
import numpy as np
import time
from smplx import SMPL
import cv2
import os
 
 
amass_npz_fname = "/home/anujithm/datasets/AMASS/ACCAD/Male1Walking_c3d/Walk_B15_-_Walk_turn_around_stageii.npz"
 
smpl_model_path = '/home/anujithm/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.pkl'
 
model = SMPL(model_path=smpl_model_path, gender="NEUTRAL", batch_size=1)
motion_data = np.load(amass_npz_fname, allow_pickle=True)
 
poses = motion_data["poses"]
betas = motion_data["betas"]
trans = motion_data.get("trans", None)
 
betas = betas[:10] if betas.ndim == 1 else betas[0, :10]
betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
 
# init first frame mesh
pose0 = torch.tensor(poses[0], dtype=torch.float32).unsqueeze(0)
out0 = model(global_orient=pose0[:, :3], body_pose=pose0[:, 3:], betas=betas_t)
verts0 = out0.vertices[0].detach().cpu().numpy()
if trans is not None:
    verts0 = verts0 + trans[0]
 
 
 
 
# 1. Initialize Open3D Mesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts0)
mesh.triangles = o3d.utility.Vector3iVector(model.faces)
mesh.compute_vertex_normals() # Ensures proper lighting
 
# 2. Setup Non-blocking Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="SMPL Open3D Renderer", width=1280, height=720)
vis.add_geometry(mesh)
 
# Optional: Add a coordinate frame for reference
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
vis.add_geometry(axis)

# Setup video writer
output_dir = "videos"
os.makedirs(output_dir, exist_ok=True)
# Extract filename from npz path without extension
video_name = os.path.splitext(os.path.basename(amass_npz_fname))[0]
output_file = os.path.join(output_dir, f"{video_name}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
mp4_writer = cv2.VideoWriter(output_file, fourcc, 30.0, (1280, 720))
frames_captured = 0
 
# 3. Animation Loop
for f in range(1, len(poses)):
    # Calculate vertices (same as your original logic)
    pose = torch.tensor(poses[f], dtype=torch.float32).unsqueeze(0)
    out = model(global_orient=pose[:, :3], body_pose=pose[:, 3:], betas=betas_t)
    verts = out.vertices[0].detach().cpu().numpy()
    if trans is not None:
        verts = verts + trans[f]
 
    # Update geometry in-place
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.compute_vertex_normals() # Re-calculate normals for the new pose
    
    # Make camera follow the character
    if trans is not None:
        character_pos = trans[f]  # Character's world position
        lookat_pos = character_pos + np.array([0, 0, 0])  # Look at character center
        
        # View from BACK at eye-level
        # Camera positioned behind character (-Y direction), at character height (Z=0)
        camera_distance = 2.0
        camera_offset = np.array([0, -camera_distance, 0])  # Behind character
        
        ctr = vis.get_view_control()
        ctr.set_lookat(lookat_pos)
        ctr.set_front(camera_offset / np.linalg.norm(camera_offset))
        ctr.set_up([0, 0, 1])  # Z-axis is up (blue towards head)
    
    # Update the visualizer
    vis.update_geometry(mesh)
    
    # Process GUI events (allows you to move the camera during animation)
    if not vis.poll_events():
        break
    vis.update_renderer()
    
    # Capture frame for video
    frame = vis.capture_screen_float_buffer(do_render=True)
    frame_np = np.asarray(frame)
    # Convert from 0-1 float to 0-255 uint8 and BGR format
    frame_bgr = (frame_np[:, :, ::-1] * 255).astype(np.uint8)
    mp4_writer.write(frame_bgr)
    frames_captured += 1

vis.destroy_window()
mp4_writer.release()
print(f"Video saved to {output_file} ({frames_captured} frames)")
 
 
# mesh = trimesh.Trimesh(vertices=verts0, faces=model.faces, process=False)
# mesh_node = pyrender.Mesh.from_trimesh(mesh)
# scene = pyrender.Scene()
# scene.add(mesh_node)
 
# # Update loop with non-blocking viewer
# viewer = pyrender.Viewer(scene, use_mesh_normals=True)
 
# for f in range(1, len(poses)):
#     pose = torch.tensor(poses[f], dtype=torch.float32).unsqueeze(0)
#     print(f'pose update: {pose.shape}')
#     out = model(global_orient=pose[:, :3], body_pose=pose[:, 3:], betas=betas_t)
#     verts = out.vertices[0].detach().cpu().numpy()
#     if trans is not None:
#         verts = verts + trans[f]
 
#     mesh.vertices = verts
#     mesh_node.mesh = trimesh.Trimesh(vertices=verts, faces=model.faces, process=False)
#     viewer.render_lock.acquire()
    
#     # 3. Update the mesh data
#     # Note: Simply changing mesh.vertices doesn't always trigger a GPU buffer update.
#     # It is often safer to recreate the mesh node or update the specific primitive.
#     scene.remove_node(mesh_node)
#     new_mesh = trimesh.Trimesh(vertices=verts, faces=model.faces, process=False)
#     mesh_node = pyrender.Mesh.from_trimesh(new_mesh)
#     scene.add(mesh_node)
    
#     # 4. Release the lock to let the viewer render the frame
#     viewer.render_lock.release()
    
#     print(f'Frame {f} updated')
#     time.sleep(1/30)
 
# # # Keep viewer alive
# # while viewer.is_active:
# #     time.sleep(0.1)