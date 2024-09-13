import torch
import toml
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from api import optimize_calibrated_multi_camera
from kornia.geometry import axis_angle_to_rotation_matrix
from tqdm import tqdm
import time

def parse_toml_camera(file_path):
    cam_data = toml.load(file_path)
    cameras = {}
    for cam_key in [key for key in cam_data if key != 'metadata']:
        cam = cam_data[cam_key]
        cameras[cam['name']] = {
            'name': cam['name'],
            'size': cam['size'],
            'matrix': cam['matrix'],
            'distortions': cam['distortions'],
            'rotation': cam['rotation'],
            'translation': cam['translation'],
            'fisheye': cam['fisheye']
        }
    return cameras

def parse_json_keypoints(parent_folder):
    keypoints_2d = {}
    camera_folders = sorted([d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))])

    for cam_folder in camera_folders:
        cam_path = os.path.join(parent_folder, cam_folder)
        json_files = sorted(glob.glob(os.path.join(cam_path, '*.json')))
        cam_keypoints = []
        for file_path in tqdm(json_files, desc=f"Loading keypoints for {cam_folder}"):
            with open(file_path, 'r') as f:
                data = json.load(f)
                people = data.get('people', [])
                if people:
                    keypoints = people[0].get('pose_keypoints_2d', [])
                    keypoints = np.array(keypoints).reshape(-1, 3)
                    cam_keypoints.append(keypoints)
                else:
                    # Handle frames with no detected people
                    cam_keypoints.append(np.zeros((26, 3)))  # Assuming 26 keypoints for halpe26
        keypoints_2d[cam_folder] = cam_keypoints
    return keypoints_2d

def parse_trc_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip the header
    header_lines = 5
    data_lines = lines[header_lines:]

    trc_data = []
    for line in tqdm(data_lines, desc="Loading 3D keypoints"):
        tokens = line.strip().split('\t')
        if len(tokens) < 2:
            continue
        # Each marker has X, Y, Z
        marker_data = np.array(tokens[2:], dtype=float).reshape(-1, 3)
        trc_data.append(marker_data)

    trc_data = np.array(trc_data)
    return trc_data

def visualize_results(X_hat, theta_hat, num_cameras):
    camera_positions = []
    for cam_idx in range(num_cameras):
        r, t = theta_hat[cam_idx].chunk(2)
        R = axis_angle_to_rotation_matrix(r[None])[0]
        cam_pos = -R.T @ t
        camera_positions.append(cam_pos.cpu().numpy())
    camera_positions = np.array(camera_positions)

    X_hat_np = X_hat.cpu().numpy()

    plt.clf()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_hat_np[:, 0], X_hat_np[:, 1], X_hat_np[:, 2], c='b', label='Optimized 3D Points', depthshade=False)
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='r', marker='^', s=100, label='Camera Positions')
    for i, pos in enumerate(camera_positions):
        ax.text(pos[0], pos[1], pos[2], f'Cam {i+1}', color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Bundle Adjustment Results')
    plt.draw()
    plt.pause(0.001)

def main():
    # **Step 1: Parse the .toml file**
    camera_file = r'C:\Users\5W555A\Desktop\0814_pilot\가마우지\calibration\Calib_scene.toml'  # Replace with your actual file path
    cameras = parse_toml_camera(camera_file)
    print("Parsed Cameras:", cameras)

    # **Step 2: Parse the .json files from multiple cameras**
    parent_folder = r'C:\Users\5W555A\Desktop\micro-bundle-adjustment\data'  # Replace with your actual parent folder path
    keypoints_2d = parse_json_keypoints(parent_folder)
    print("Parsed 2D Keypoints from multiple cameras")

    # **Step 3: Parse the .trc file**
    trc_file = r'C:\Users\5W555A\Desktop\micro-bundle-adjustment\data\test.trc'  # Replace with your actual file path
    keypoints_3d = parse_trc_file(trc_file)
    print("Parsed 3D Keypoints:", keypoints_3d.shape)

    # **Step 4: Match the 2D and 3D keypoints for each camera**

    # Define keypoints to use and their mappings
    keypoint_ids_to_use = [
        0, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    ]
    global_keypoint_indices = {keypoint_id: idx for idx, keypoint_id in enumerate(keypoint_ids_to_use)}

    keypoint_id_to_trc_index = {
        0: 15,   # 'nose' -> 'Nose'
        5: 19,   # 'left_shoulder' -> 'LShoulder'
        6: 16,   # 'right_shoulder' -> 'RShoulder'
        7: 20,   # 'left_elbow' -> 'LElbow'
        8: 17,   # 'right_elbow' -> 'RElbow'
        9: 21,   # 'left_wrist' -> 'LWrist'
        10: 18,  # 'right_wrist' -> 'RWrist'
        11: 7,   # 'left_hip' -> 'LHip'
        12: 1,   # 'right_hip' -> 'RHip'
        13: 8,   # 'left_knee' -> 'LKnee'
        14: 2,   # 'right_knee' -> 'RKnee'
        15: 9,   # 'left_ankle' -> 'LAnkle'
        16: 3,   # 'right_ankle' -> 'RAnkle'
        17: 14,  # 'head' -> 'Head'
        18: 13,  # 'neck' -> 'Neck'
        19: 0,   # 'hip' -> 'Hip'
        20: 10,  # 'left_big_toe' -> 'LBigToe'
        21: 4,   # 'right_big_toe' -> 'RBigToe'
        22: 11,  # 'left_small_toe' -> 'LSmallToe'
        23: 5,   # 'right_small_toe' -> 'RSmallToe'
        24: 12,  # 'left_heel' -> 'LHeel'
        25: 6    # 'right_heel' -> 'RHeel'
    }

    N_keypoints = len(global_keypoint_indices)
    X_0_global = torch.zeros((N_keypoints, 3), dtype=torch.float32)
    keypoint_counts = torch.zeros(N_keypoints, dtype=torch.int)

    observations = []
    camera_names = sorted(cameras.keys())
    num_cameras = len(camera_names)

    # Prepare initial camera parameters
    theta_0_list = []

    for cam_idx, cam_name in enumerate(camera_names):
        cam_keypoints_2d = keypoints_2d.get(cam_name)
        if cam_keypoints_2d is None:
            print(f"No keypoints found for camera {cam_name}")
            continue
        cam = cameras[cam_name]

        # Initial camera parameters
        r_0 = torch.tensor(cam['rotation'], dtype=torch.float32)
        t_0 = torch.tensor(cam['translation'], dtype=torch.float32)
        theta_0 = torch.cat((r_0, t_0), dim=-1)
        theta_0_list.append(theta_0)

    # Concatenate theta_0
    theta_0 = torch.stack(theta_0_list, dim=0)  # Shape: (num_cameras, 6)

    # **Batch Processing**
    batch_size = 100  # Adjust based on your system's capacity
    total_frames = len(keypoints_3d)
    num_batches = (total_frames + batch_size - 1) // batch_size

    # Start real-time visualization
    plt.ion()
    for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
        start_frame = batch_idx * batch_size
        end_frame = min(start_frame + batch_size, total_frames)

        batch_observations = []
        keypoint_counts_batch = torch.zeros(N_keypoints, dtype=torch.int)
        X_0_batch = torch.zeros((N_keypoints, 3), dtype=torch.float32)

        for cam_idx, cam_name in enumerate(camera_names):
            cam_keypoints_2d = keypoints_2d.get(cam_name)
            if cam_keypoints_2d is None:
                continue
            cam = cameras[cam_name]

            for frame_idx in range(start_frame, end_frame):
                if frame_idx >= len(cam_keypoints_2d):
                    break
                kp_2d = cam_keypoints_2d[frame_idx]
                kp_3d = keypoints_3d[frame_idx]

                x_im_list = []
                inds_list = []

                for keypoint_id in keypoint_ids_to_use:
                    global_idx = global_keypoint_indices[keypoint_id]

                    # Get 2D keypoint
                    if keypoint_id >= kp_2d.shape[0]:
                        continue
                    kp_2d_point = kp_2d[keypoint_id]
                    confidence = kp_2d_point[2]
                    if confidence < 0.1:
                        continue

                    # Get corresponding 3D keypoint
                    trc_index = keypoint_id_to_trc_index.get(keypoint_id)
                    if trc_index is None:
                        continue
                    if trc_index >= kp_3d.shape[0]:
                        continue
                    kp_3d_point = kp_3d[trc_index]

                    # Update batch X_0
                    X_0_batch[global_idx] += torch.tensor(kp_3d_point, dtype=torch.float32)
                    keypoint_counts_batch[global_idx] += 1

                    # Add to observation
                    x_im_list.append(kp_2d_point[:2])
                    inds_list.append(global_idx)

                if len(x_im_list) == 0:
                    continue
                x_im = torch.tensor(np.array(x_im_list), dtype=torch.float32)
                inds = torch.tensor(np.array(inds_list), dtype=torch.long)

                # Add to observations
                batch_observations.append({
                    'x_im': x_im,
                    'inds': inds,
                    'camera_idx': cam_idx
                })

        # Average the batch X_0
        for i in range(N_keypoints):
            if keypoint_counts_batch[i] > 0:
                X_0_batch[i] /= keypoint_counts_batch[i]
            else:
                # Handle keypoints not observed at all
                X_0_batch[i] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        # Use the batch X_0 for optimization
        X_hat_batch, theta_hat_batch = optimize_calibrated_multi_camera(
            X_0_batch,
            batch_observations,
            theta_0,
            num_cameras=num_cameras,
            num_steps=50  # Adjust as needed
        )

        # Optionally update global estimates or visualize
        visualize_results(X_hat_batch, theta_hat_batch, num_cameras)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
