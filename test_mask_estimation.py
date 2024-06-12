import numpy as np
import cv2
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def estimateMaskPosition(poses, pose_idx, model_pcd, K, use_last_frames = 10):

    rot_mats = poses[pose_idx - use_last_frames: pose_idx,:3,:3]
    trans_vecs = poses[pose_idx - use_last_frames: pose_idx,:3,3]
    # Convert the matrix to Euler angles (in degrees)
    r = R.from_matrix(rot_mats)
    angles = r.as_euler("zyx",)
    rot_velocities = []
    for idx, t_vec in enumerate(angles):
        if idx > 0:
            vel = t_vec - angles[idx-1]
            rot_velocities.append(vel)
    trans_velocities = []
    for idx, t_vec in enumerate(trans_vecs):
        if idx > 0:
            vel = t_vec - trans_vecs[idx-1]
            trans_velocities.append(vel)

    average_rot_vel = np.average(rot_velocities, axis = 0)
    average_trans_vel = np.average(trans_velocities, axis = 0)


    est_angle = angles[-1] + average_rot_vel
    est_rot_mat = R.from_euler("zyx", est_angle).as_matrix()
    est_trans_vec = trans_vecs[-1] + average_trans_vel

    est_pose = np.identity(4)
    est_pose[:3,:3] = est_rot_mat
    est_pose[:3,3] = est_trans_vec

    print(est_pose)
    print(poses[pose_idx])


def compare_masks(mask_gt,mask_est):
    mask_gt_bool = mask_gt > 0
    mask_est_bool = mask_est > 0 
    pixel_match = np.logical_and(mask_gt_bool, mask_est_bool)
    pixel_total_gt = len(mask_gt[mask_gt_bool])
    return len(pixel_match[pixel_match]) / pixel_total_gt

def loadPoses(pose_dir):
    pose_paths = os.listdir(pose_dir)
    pose_paths.sort()
    rot_movements = [0]
    poses = []
    for idx,pose_file in enumerate(pose_paths):
        pose = None
        if pose_file.endswith(".txt"):
            pose = np.loadtxt(os.path.join(pose_dir, pose_file))
        elif pose_file.endswith(".npy"):
            pose = np.load(os.path.join(pose_dir, pose_file))
        else:
            continue
        poses.append(pose)
    return np.array(poses).reshape((-1,4,4))

if __name__ == "__main__":
    K_path = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/cam_K.txt"

    K = np.loadtxt(K_path)
    model = o3d.io.read_point_cloud("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/model_icp.ply")
    poses = loadPoses(pose_dir="/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/pose")
    estimateMaskPosition(poses= poses, pose_idx= 80,model_pcd=model, K =K, use_last_frames= 20)