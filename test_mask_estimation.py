import numpy as np
import cv2
import os
import trimesh
from scipy.spatial.transform import Rotation as R

def estimateMaskPosition(poses, pose_idx, model, K, resolution, use_last_frames = 10):


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
    points_2d = get_points_2d(est_pose,K,model.vertices.copy())
    points_2d = points_2d.astype(np.int16)
    mask_est = np.zeros(resolution)
    mask_est[points_2d[:,0],points_2d[:,1]] = 255
    #close gaps
    kernel = np.ones((2, 2), np.uint8)
    mask_est = cv2.dilate(mask_est, kernel, iterations=1)
    mask_est = cv2.erode(mask_est, kernel, iterations=1)

    return mask_est
    #model_pcd.transform(est_pose)

def get_points_2d(T, K, points):
    # unit is m
    #T = np.identity(4)
    #T[:3,3] = np.array([0.083,-0.3442,-1.097])
    rot_mat = T[:3,:3]
    rotV, _ = cv2.Rodrigues(rot_mat)
    tVec = T[:3,3]
    points_2d, _ = cv2.projectPoints(points, rotV, tVec, K, (0, 0, 0, 0))
    points_2d = points_2d.reshape(-1,2)
    return points_2d

def draw_axis(img, T, K):
    # unit is m
    #T = np.identity(4)
    #T[:3,3] = np.array([0.083,-0.3442,-1.097])
    rot_mat = T[:3,:3]
    rotV, _ = cv2.Rodrigues(rot_mat)
    tVec = T[:3,3]
    points = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, tVec, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(np.array(axisPoints[3].ravel(), dtype=np.int16)), tuple(np.array(axisPoints[0].ravel(), dtype=np.int16)), (0,0,255), 3)
    img = cv2.line(img, tuple(np.array(axisPoints[3].ravel(), dtype=np.int16)), tuple(np.array(axisPoints[1].ravel(), dtype=np.int16)), (0,255,0), 3)
    img = cv2.line(img, tuple(np.array(axisPoints[3].ravel(), dtype=np.int16)), tuple(np.array(axisPoints[2].ravel(), dtype=np.int16)), (255,0,0), 3)
    return img

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
    model = trimesh.load("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/model.ply")#o3d.io.read_point_cloud("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/model_icp.ply")
    poses = loadPoses(pose_dir="/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/pose")
    mask_est = estimateMaskPosition(poses= poses, pose_idx= 80,model=model, K =K, resolution= (720, 1280), use_last_frames= 20)
    mask_gt = cv2.imread("")
    pose_img = cv2.imread("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/rgb/00079.png")
    pose_img = draw_axis(pose_img, poses[80], K)
    cv2.imwrite("mask_prediction/out_mask.jpg", mask_est)
    cv2.imwrite("mask_prediction/out_pose.jpg", pose_img)