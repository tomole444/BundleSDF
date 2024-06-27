import numpy as np
import os
import trimesh
import cv2
#from Utils import *
import matplotlib.pyplot as plt


class BenchmarkADD:

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


class BenchmarkSegmentation:

    def __init__(self, masks_est_dir, masks_gt_dir) -> None:
        self.masks_est_dir = masks_est_dir
        self.masks_gt_dir = masks_gt_dir
        
        
        self.mask_est_paths = os.listdir(masks_est_dir)
        self.mask_est_paths.sort()
        self.mask_gt_paths = os.listdir(masks_gt_dir)
        self.mask_gt_paths.sort()

        if len(self.mask_gt_paths) != len(self.mask_est_paths):
            raise RuntimeError("Masks dont line up")
        
        self.iou_arr = []
        self.pixel_acc_arr = []
        self.precision_arr = []
        self.recall_arr = []

        self.ids = []
    
    @staticmethod
    def calc_iou( mask_est, mask_gt):
        intersection = np.logical_and(mask_est, mask_gt)
        union = np.logical_or(mask_est, mask_gt)
        if np.sum(union) != 0:
            # print (f"sum1 {np.sum(intersection)} sum2 {np.sum(union)}")
            iou = np.sum(intersection) / np.sum(union)
        else:
            iou = -1
        # print(iou)
        return iou
    
    @staticmethod
    def calc_pixel_acc(mask_est, mask_gt):
        confusion = BenchmarkSegmentation.calc_confusion(mask_est, mask_gt)
        denominator = (confusion["tp"] + confusion["tn"] + confusion["fp"] + confusion["fn"])
        if denominator != 0:
            pixel_acc = (confusion["tp"] + confusion["tn"]) / denominator
        else:
            pixel_acc = -1
        return pixel_acc
    
    @staticmethod
    def calc_precision(mask_est, mask_gt):
        confusion = BenchmarkSegmentation.calc_confusion(mask_est, mask_gt)
        denominator = (confusion["tp"] + confusion["fp"])
        if denominator != 0:
            precision = (confusion["tp"]) / denominator
        else:
            precision = -1
        return precision
    
    @staticmethod
    def calc_recall(mask_est, mask_gt):
        confusion = BenchmarkSegmentation.calc_confusion(mask_est, mask_gt)
        denominator = (confusion["tp"] + confusion["fn"])
        if denominator != 0:
            recall = (confusion["tp"]) / denominator
        else:
            recall = -1
        return recall

    @staticmethod
    def calc_confusion(mask_est, mask_gt):
        tp = np.sum(np.logical_and(mask_est, mask_gt))
        tn = np.sum(np.logical_and(np.logical_not(mask_est), np.logical_not(mask_gt)))
        fp = np.sum(np.logical_and(mask_est, np.logical_not(mask_gt)))
        fn = np.sum(np.logical_and(np.logical_not(mask_est), mask_gt))
        confusion = {
                    "tp":tp,
                    "tn":tn,
                    "fp":fp,
                    "fn":fn
                    }
        return confusion

    def calc_metrics(self):
        for idx, mask_est_path in enumerate(self.mask_est_paths):
            print(f"calulating metrics for {mask_est_path}")
            mask_est_path = os.path.join(self.masks_est_dir, mask_est_path)
            mask_est = cv2.imread(mask_est_path, cv2.IMREAD_GRAYSCALE)

            mask_gt_path = os.path.join(self.masks_gt_dir, self.mask_gt_paths[idx])
            mask_gt = cv2.imread(mask_gt_path, cv2.IMREAD_GRAYSCALE)

            mask_est = np.where(mask_est >= 1, 1, 0)
            mask_gt = np.where(mask_gt >= 1, 1, 0)

            # if idx == 160:
            #     print("te")
            iou = BenchmarkSegmentation.calc_iou(mask_est, mask_gt)
            pixel_acc = BenchmarkSegmentation.calc_pixel_acc(mask_est, mask_gt)
            precision = BenchmarkSegmentation.calc_precision(mask_est, mask_gt)
            recall = BenchmarkSegmentation.calc_recall(mask_est, mask_gt)

            self.iou_arr.append(iou)
            self.pixel_acc_arr.append(pixel_acc)
            self.precision_arr.append(precision)
            self.recall_arr.append(recall)
            
            self.ids.append(idx)
        
        self.iou_arr = np.array(self.iou_arr)
        self.pixel_acc_arr = np.array(self.pixel_acc_arr)
        self.precision_arr = np.array(self.precision_arr)
        self.recall_arr = np.array(self.recall_arr)
        self.ids = np.array(self.ids)
    
    def plot_results(self):
        ax = plt.gca()
        #ax.set_ylim([0, 1])
        
        plt.plot(self.ids, self.iou_arr, label = "IOU")
        plt.plot(self.ids, self.pixel_acc_arr, label = "Pixel Acc")
        plt.plot(self.ids, self.precision_arr, label = "Precision")
        plt.plot(self.ids, self.recall_arr, label = "Recall")

        plt.legend(loc="upper right")
        ax.set_title(self.masks_est_dir, fontsize = 32, fontweight ='bold')


        plt.show()

    def save_results(self, path):
        save_arr = dict()
        save_arr["iou"] = self.iou_arr 
        save_arr["pixel_acc"] = self.pixel_acc_arr 
        save_arr["precision"] = self.precision_arr 
        save_arr["recall"] = self.recall_arr 

        save_arr["ids"] = self.ids
        save_arr = np.array(save_arr, dtype=object)
        np.save(path, save_arr, allow_pickle=True)


def calcADD():
    bench = BenchmarkADD(pose_pred_dir="/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuchVideoPVNetSegOnly/ob_in_cam",
                      pose_gt_dir= "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/pose",
                      model_path="/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/model.ply",
                      model_diameter=0.211,
                      first_pose_adjust= False) 
    bench.run_add_pose()
    #bench.run_occlusion()
    bench.plot_results()
    bench.save_results("benchmarks/BuchVideo/ADD_BundleSDF_pvnet_segmentation_only.npy")

def calcMaskMetrics():
    bench = BenchmarkSegmentation(masks_est_dir= "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/masks_xmem_first_pvnet",
                                  masks_gt_dir="/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/masks")
    bench.calc_metrics()
    bench.plot_results()
    bench.save_results("benchmarks/BuchVideo/mask_analysis/Metrics_first_mask_pvnet_xmem.npy")


if __name__ == "__main__":
    #calcADD()
    calcMaskMetrics()