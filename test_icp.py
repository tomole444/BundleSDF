import cupoch as cph
import numpy as np
import os
import cv2
import open3d as o3d

def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    #print(zs.min())
    #print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts

if __name__ == "__main__":
    model_path = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/model_icp.ply"
    depth_path = "outBuchVideoLimitMovement/depth/00000.png"
    mask_path = "outBuchVideoLimitMovement/mask/00000.png"
    pose_path = "outBuchVideoLimitMovement/ob_in_cam/00000.txt"
    pose_icp_path = "/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuchTest/ob_in_cam/00000.txt"
    pose_gt_path = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/pose/00000.npy"
    K_path = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/cam_K.txt"

    K = np.loadtxt(K_path)
    pose = np.loadtxt(pose_path)
    pose_icp = np.loadtxt(pose_icp_path)
    pose_gt = np.load(pose_gt_path)
    print(cph.utility.is_cuda_available())
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    sem_out = np.where(mask_img>0.8,1,0)
    #sem_out = np.where(radial_out<=max_radii_dm[keypoint_count-1], sem_out,0)
    depth_map = depth_img * sem_out
    xyz_mm = rgbd_to_point_cloud(K,depth_map)
    xyz_m = xyz_mm / 1e3
    depth_scene_pcd = o3d.geometry.PointCloud()
    depth_scene_pcd.points = o3d.utility.Vector3dVector(xyz_m)#cph.utility.Vector3fVector(xyz_mm)

    model_pcd = o3d.io.read_point_cloud(model_path)
    model_pcd_2 = o3d.io.read_point_cloud(model_path)
    model_pcd_3 = o3d.io.read_point_cloud(model_path)
    #model_pcd.transform(pose)

    model_pcd.paint_uniform_color([0, 0.651, 0.929])
    model_pcd_2.paint_uniform_color([0, 1, 0])
    model_pcd_3.paint_uniform_color([1, 0, 0])
    depth_scene_pcd.paint_uniform_color([1, 0.706, 0])
    depth_scene_pcd.estimate_normals()
    model_pcd.estimate_normals()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria()
    threshold = 0.005
    trans_init = pose
    reg_p2p = o3d.pipelines.registration.registration_icp(
            model_pcd, depth_scene_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria)
    
    reg_p2l = o3d.pipelines.registration.registration_icp(
            model_pcd, depth_scene_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria)
    model_pcd.transform(reg_p2l.transformation)
    model_pcd_2.transform(pose_gt)
    model_pcd_3.transform(pose_icp)
    o3d.visualization.draw_geometries([model_pcd, depth_scene_pcd])