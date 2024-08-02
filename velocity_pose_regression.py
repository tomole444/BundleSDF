import numpy as np
import cv2
import os
import trimesh
import time
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import LinearRegression



class VelocityPoseRegression:
    def __init__(self, K, model_path, cfg):
        self.K = K
        self.model = trimesh.load(model_path)#o3d.io.read_point_cloud("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/model_icp.ply")

        self.est_pose = np.identity(4)
        self.cfg = cfg

        self.mask_est = None
        self.pose_data = {"tfs": [], "vels": {"quat": [], "trans": []}, "accs": {"quat": [], "trans": []}, "time_stamps": []}

    def predictPose(self):
        use_last = self.cfg["estimation"]["use_last"] -1 # -1 because index is one lower than count 
        use_time_relation = self.cfg["estimation"]["use_time_relation"]
        tf_count = len(self.pose_data["tfs"])
        vel_quat_count = len(self.pose_data["vels"]["quat"])
        vel_trans_count = len(self.pose_data["vels"]["trans"])
        time_stamp_count = len(self.pose_data["time_stamps"])


        tf_lower_index = tf_count - use_last if tf_count - use_last >= 0 else 0 
        vel_quat_lower_index = vel_quat_count - use_last if vel_quat_count - use_last >= 0 else 0
        vel_trans_lower_index = vel_trans_count - use_last if vel_trans_count - use_last >= 0 else 0
        time_stamp_lower_index = time_stamp_count - (vel_quat_count - vel_quat_lower_index)

        last_poses = np.array(self.pose_data["tfs"]) [tf_lower_index : tf_count]
        last_vels_quat = np.array(self.pose_data["vels"]["quat"]) [vel_quat_lower_index : vel_quat_count]
        last_vels_trans = np.array(self.pose_data["vels"]["trans"]) [vel_trans_lower_index : vel_trans_count]
        last_time_stamps = np.array(self.pose_data["time_stamps"]) [time_stamp_lower_index : time_stamp_count]

        x = last_time_stamps if use_time_relation else np.arange(len(last_vels_quat))
        last_rot_mat = last_poses[-1, :3, :3].reshape((3,3))
        last_trans_vec = last_poses[-1, :3, 3].reshape((3,1))

        #Convert the matrix to Euler angles
        r = R.from_matrix(last_rot_mat)
        last_angles = r.as_quat()


        #create linear regression
        x_stacked = np.vstack((x,x,x)).T
        linreg_rot = LinearRegression()
        linreg_rot.fit(x_stacked, last_vels_quat) 
        linreg_trans = LinearRegression()
        linreg_trans.fit(x_stacked, last_vels_trans) 

        #predict vels 
        
        x_pred = np.array([time.time(), time.time(), time.time()]).reshape(-1,3) if use_time_relation else np.array([len(x), len(x), len(x)]).reshape(-1,3)
        est_quat_vel = linreg_rot.predict(x_pred)
        est_trans_vel = linreg_trans.predict(x_pred)

        time_diff = time.time() - last_time_stamps[-1]
        est_quat = last_angles + est_quat_vel * time_diff 
        est_rot_mat = R.from_quat(est_quat).as_matrix() 
        est_trans_vec = last_trans_vec + est_trans_vel 

        est_pose = np.identity(4)
        est_pose[:3,:3] = est_rot_mat
        est_pose[:3,3] = est_trans_vec

        self.est_pose = est_pose
        return est_pose

    def calcPoseMask(self, resolution, force_pose = None):
        pose = self.est_pose
        if force_pose is not None:
            pose = force_pose
        model_point_in_model_coords = self.model.vertices.copy()
        points_2d = VelocityPoseRegression.get_points_2d(pose,self.K,model_point_in_model_coords)
        points_2d = points_2d.astype(np.int16)
        mask_est = np.zeros(resolution)
        mask_est[points_2d[:,1],points_2d[:,0]] = 255
        #close gaps
        kernel = np.ones((2, 2), np.uint8)
        mask_est = cv2.dilate(mask_est, kernel, iterations=1)
        mask_est = cv2.erode(mask_est, kernel, iterations=1)
        if force_pose is None:
            self.mask_est = mask_est
        return mask_est
    
    def evaluateMask(self, mask_gt):
        mask_gt_bool = mask_gt > 0
        mask_est_bool = self.mask_est > 0 
        pixel_correct = np.logical_and(mask_gt_bool, mask_est_bool)
        #pixel_false_positive = np.logical_xor(mask_gt_bool, mask_est_bool)
        pixel_total_est = len(mask_est[mask_est_bool])
        pixel_correct_count = len(pixel_correct[pixel_correct])

        error = pixel_correct_count / pixel_total_est#valid_pixel / pixel_total_gt

        err_mask_viz = np.where(pixel_correct, 255, 0)
        return  error, err_mask_viz
    
    def evaluatePose(self, pose_gt):
        err = np.sum(np.abs(self.est_pose - poses[pose_idx]))
        print("estimated: ", self.est_pose)
        print("gt:", pose_gt)
        print("pose - err: ", err)
        return err
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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
    resolution = (720, 1280)

    regresser = VelocityPoseRegression(K,"/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/model_icp.ply")
    poses = VelocityPoseRegression.loadPoses(pose_dir="/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/pose")
    pose_idx= 80
    use_last_frames= 20
    last_poses = poses[pose_idx - use_last_frames: pose_idx,:,:]
    pose_gt = poses[pose_idx]
    est_pose = regresser.predictPose(last_poses)
    err_pose = regresser.evaluatePose(pose_gt)
    mask_est = regresser.calcPoseMask(resolution)
    

    mask_gt = regresser.calcPoseMask(resolution, force_pose = pose_gt)#cv2.imread("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/masks/00080.png", cv2.IMREAD_GRAYSCALE)
    mask_gt_occ = cv2.imread("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/masks/00080.png", cv2.IMREAD_GRAYSCALE)
    err_mask, viz1 = regresser.evaluateMask(mask_gt)
    err_mask_occ, viz2 = regresser.evaluateMask(mask_gt_occ)
    print("Mask Valid GT:", err_mask)
    print("Mask Valid Occ:", err_mask_occ)
    
    
    pose_img = cv2.imread("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/rgb/00080.png")
    pose_img = VelocityPoseRegression.draw_axis(pose_img, est_pose, K)
    
    cv2.imwrite("mask_prediction/out_mask_est.jpg", mask_est)
    cv2.imwrite("mask_prediction/out_mask_gt.jpg", mask_gt)
    #cv2.imwrite("mask_prediction/out_mask_err.jpg", viz1)
    cv2.imwrite("mask_prediction/out_pose.jpg", pose_img)