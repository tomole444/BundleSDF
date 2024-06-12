import numpy as np
import cv2
import os
import o3d

def estimateMaskPosition(poses, pose_idx):
    


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
    return poses

if __name__ == "__main__":
    model = o3d.io.read_point_cloud("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/model_icp.ply")
    poses = loadPoses(pose_dir="/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/pose")