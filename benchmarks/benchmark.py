import numpy as np
import os
import trimesh
import cv2
#from Utils import *
import matplotlib.pyplot as plt


class Benchmark:

    def __init__(self, pose_pred_dir, pose_gt_dir, model_path, model_diameter, first_pose_adjust = True):
        self.pose_pred_dir = pose_pred_dir
        self.pose_gt_dir = pose_gt_dir
        self.model_diameter = model_diameter

        self.model_path = model_path
        self.mesh = trimesh.load(self.model_path)

        self.first_pose_adjust = first_pose_adjust

        self.add_errs = []
        self.trans_errs = []
        self.rot_errs = []
        self.gt_poses = []
        self.pred_poses = []
        self.ids = []


        self.result_y = []
        

    def run_add_pose(self):

        #preditcted poses
        pose_pred_files = os.listdir(self.pose_pred_dir)
        pose_pred_files.sort()
        shape = (4,4)
        for idx,pose_file in enumerate(pose_pred_files):
            pose = None
            if pose_file.endswith(".txt"):
                pose = np.loadtxt(os.path.join(self.pose_pred_dir, pose_file))
            elif pose_file.endswith(".npy"):
                pose = np.load(os.path.join(self.pose_pred_dir, pose_file))
            else:
                continue
            if(pose.shape != shape):
                print(f"new shape {pose.shape} at {idx} -> forgetting last row")
                shape = pose.shape
                pose = pose[:4, :4]
            self.pred_poses.append(pose)
            self.ids.append(idx)

        self.pred_poses = np.array(self.pred_poses)
        self.ids = np.array(self.ids)

        # ground truth poses

        pose_gt_files = os.listdir(self.pose_gt_dir)
        pose_gt_files.sort()
        for idx, pose_file in enumerate(pose_gt_files):
            self.gt_poses.append(np.load(os.path.join(self.pose_gt_dir, pose_file)))
        self.gt_poses = np.array(self.gt_poses)
        

        #print(pred_poses.shape)
        #Align first frame to match gt frame 
        if self.first_pose_adjust:
            self.pred_poses = self.pred_poses@np.linalg.inv(self.pred_poses[0])@self.gt_poses[0]

        

        for i in range(len(self.pred_poses)):
            #adi = adi_err(pred_poses[i],gt_poses[i],mesh.vertices.copy())
            add, trans_err, rot_err = self.calc_add_error(self.pred_poses[i],self.gt_poses[i])
            #adi_errs.append(adi)
            self.add_errs.append(add)
            self.trans_errs.append(trans_err)
            self.rot_errs.append(rot_err)

        #adi_errs = np.array(adi_errs)
        self.add_errs = np.array(self.add_errs)
        self.trans_errs = np.array(self.trans_errs)
        self.rot_errs = np.array(self.rot_errs)
        self.result_y = self.add_errs.copy()

        mask = []
        # ignore poses with identitidy matrix
        for idx,pose in enumerate(self.pred_poses):
            if pose.round(decimals=6)[2,3] < 0.001 :
                mask.append(False)
            else:
                mask.append(True)
        
        calced_add = self.add_errs[mask]
        good_add = calced_add[calced_add < (0.1 * self.model_diameter)]
        if len(calced_add) > 0:
            print("ADD = ", len(good_add) / len(calced_add))
        else:
            print("ADD = 0")

        #ADDS_AUC = compute_auc(adi_errs)*100
        #ADD_AUC = compute_auc(add_errs)*100
    
    def run_occlusion(self):
        mask_full_paths = os.listdir(self.pose_gt_dir)
        mask_full_paths.sort()
        mask_full_counts = []
        for mask_full_path in mask_full_paths:
            mask_full = cv2.imread(os.path.join(self.pose_gt_dir, mask_full_path), cv2.IMREAD_GRAYSCALE)
            mask_full = mask_full.ravel()
            mask_full_counts.append(len(mask_full[mask_full > 0]))
        mask_full_counts = np.array(mask_full_counts)

        mask_occ_paths = os.listdir(self.pose_pred_dir)
        mask_occ_paths.sort()
        mask_occ_counts = []
        for idx, mask_occ_path in enumerate(mask_occ_paths):
            mask_occ = cv2.imread(os.path.join(self.pose_pred_dir, mask_occ_path), cv2.IMREAD_GRAYSCALE)
            mask_occ =mask_occ.ravel()
            mask_occ_counts.append(len(mask_occ[mask_occ > 0]))
            self.ids.append(idx)
        mask_occ_counts = np.array(mask_occ_counts)
        self.ids = np.array(self.ids)

        self.occ_percentage = np.divide(mask_occ_counts, mask_full_counts)
        self.result_y = self.occ_percentage.copy()


    def plot_results(self):
        y = self.result_y

        x = self.ids
        ax = plt.gca()
        ax.set_ylim([0, 1])
        plt.plot(x,y)
        plt.show()

    def save_results(self, path):
        save_arr = dict()
        save_arr["result_y"] = self.result_y 
        save_arr["trans_err"] = self.trans_errs
        save_arr["rot_err"] = self.rot_errs
        save_arr["ids"] = self.ids
        save_arr = np.array(save_arr, dtype=object)
        np.save(path,save_arr, allow_pickle=True)
    
    def load_results_add(self, path):
        load_arr = np.load(path, allow_pickle=True).item()
        self.add_errs = load_arr["result_y"]
        self.ids = load_arr["ids"]

    def calc_add_error(self, T_pred, T_gt):
        model_points = self.mesh.vertices.copy()
        model_points_hom = np.concatenate((model_points, np.ones((model_points.shape[0],1))),axis=-1)
        projected_model_points_pred = (T_pred@model_points_hom.T).T[:,:3]
        projected_model_points_gt = (T_gt@model_points_hom.T).T[:,:3]

        T_pred_trans = np.identity(4)
        T_pred_trans[:3,3] = T_pred[:3,3]
        T_gt_trans = np.identity(4)
        T_gt_trans[:3,3] = T_gt[:3,3]
        projected_model_points_pred_trans = (T_pred_trans@model_points_hom.T).T[:,:3]
        projected_model_points_gt_trans = (T_gt_trans@model_points_hom.T).T[:,:3]

        T_pred_rot = np.identity(4)
        T_pred_rot[:3,:3] = T_pred[:3,:3]
        T_gt_rot = np.identity(4)
        T_gt_rot[:3,:3] = T_gt[:3,:3]
        projected_model_points_pred_rot = (T_pred_rot@model_points_hom.T).T[:,:3]
        projected_model_points_gt_rot = (T_gt_rot@model_points_hom.T).T[:,:3]

        add_err = np.linalg.norm(projected_model_points_gt - projected_model_points_pred, axis=1).mean()
        trans_err = np.linalg.norm(projected_model_points_gt_trans - projected_model_points_pred_trans, axis=1).mean()
        rot_err = np.linalg.norm(projected_model_points_gt_rot - projected_model_points_pred_rot, axis=1).mean()

        return add_err, trans_err, rot_err


if __name__ == "__main__":
    bench = Benchmark(pose_pred_dir="/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo2/outPVNet239/pose",
                      pose_gt_dir= "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo2/pose",
                      model_path="/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo2/model.ply",
                      model_diameter=0.211,
                      first_pose_adjust= False) 
    bench.run_add_pose()
    #bench.run_occlusion()
    bench.plot_results()
    bench.save_results("benchmarks/BuchVideo2/ADD_PVNet_Big_dataset.npy")