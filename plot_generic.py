import numpy as np
import matplotlib.pyplot as plt
import os
from math import dist

import time


class ResultPlotter:
    def __init__(self):
        load_arr = np.load("/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/benchmarks/BuchVideo/ADD_PVNet_orig.npy", allow_pickle=True).item()
        self.add_pvnet_orig = load_arr["result_y"]

        pose_dir = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/outPVNet239/pose"
        self.mask = ResultPlotter.calcMask(pose_dir=pose_dir)
        
        self.x = load_arr["ids"]
        self.x_masked = self.x[self.mask]


        self.add_pvnet_orig = self.add_pvnet_orig[self.mask]
        load_arr = np.load("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/outPVNet239_temp/confidences_indiv.npy", allow_pickle=True).item()
        cov_invs = load_arr["result_y"] 
        confidence_sum = np.sum(np.abs(cov_invs), axis=2)
        #confidence_sum = np.sum(confidence_sum, axis=2)
        confidence_sum = 1-(confidence_sum / 5)
        confidence_kpt_0 = confidence_sum[:,0] #1 - (confidence_sum[:,0]/ 5)
        confidence_kpt_1 = confidence_sum[:,1] #1 - (confidence_sum[:,1]/ 5)
        confidence_kpt_2 = confidence_sum[:,2] #1 - (confidence_sum[:,2]/ 5)
        confidence_kpt_3 = confidence_sum[:,3] #1 - (confidence_sum[:,3]/ 5)
        confidence_kpt_4 = confidence_sum[:,4] #1 - (confidence_sum[:,4]/ 5)
        confidence_kpt_5 = confidence_sum[:,5] #1 - (confidence_sum[:,5]/ 5)
        confidence_kpt_6 = confidence_sum[:,6] #1 - (confidence_sum[:,6]/ 5)
        confidence_kpt_7 = confidence_sum[:,7] #1 - (confidence_sum[:,7]/ 5)
        confidence_kpt_8 = confidence_sum[:,8] #1 - (confidence_sum[:,8]/ 5)
        confidence_kpt_0 = confidence_kpt_0[self.mask]
        confidence_kpt_1 = confidence_kpt_1[self.mask]
        confidence_kpt_2 = confidence_kpt_2[self.mask]
        confidence_kpt_3 = confidence_kpt_3[self.mask]
        confidence_kpt_4 = confidence_kpt_4[self.mask]
        confidence_kpt_5 = confidence_kpt_5[self.mask]
        confidence_kpt_6 = confidence_kpt_6[self.mask]
        confidence_kpt_7 = confidence_kpt_7[self.mask]
        confidence_kpt_8 = confidence_kpt_8[self.mask]

        confidence_sum_no_last = confidence_sum[:,:-1]
        self.stabw = np.std(confidence_sum_no_last,axis = 1)[self.mask]
        #stabw = np.sqrt(stabw)
        self.avg = np.average(confidence_sum_no_last,axis = 1)[self.mask]



        load_arr = np.load("/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/benchmarks/BuchVideo/ADD_PVNet_upnp.npy", allow_pickle=True).item()
        self.add_pvnet_upnp = load_arr["result_y"]
        self.add_pvnet_upnp = self.add_pvnet_upnp[self.mask]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_orig.npy", allow_pickle=True).item()
        self.add_bundle_orig = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_PeriodicPVNet_orig.npy", allow_pickle=True).item()
        self.add_bundle_periodic_orig = load_arr["result_y"]
        
        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_PeriodicPVNet_upnp.npy", allow_pickle=True).item()
        self.add_bundle_periodic_upnp = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_LimitRot.npy", allow_pickle=True).item()
        self.add_bundle_limit_rot = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_LimitRotTrans.npy", allow_pickle=True).item()
        self.add_bundle_limit_rot_trans = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_ICP.npy", allow_pickle=True).item()
        self.add_bundle_icp = load_arr["result_y"]
        self.mask = ResultPlotter.calcMask(pose_dir="/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuchVideoICP/ob_in_cam")
        self.add_bundle_icp_masked = self.add_bundle_icp[self.mask]
        self.x_masked = self.x[self.mask]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_Occlusion_Aware.npy", allow_pickle=True).item()
        self.add_bundle_occ_aware = load_arr["result_y"]


        #rot_movement = np.load("outBuchVideoNoLimiting/rot_movement/1699.npy", allow_pickle=True)

        #trans_movement = np.load("outBuchVideoNoLimiting/trans_movement/1699.npy", allow_pickle=True)
        
        self.rot_movement_2 = ResultPlotter.calcRotMovement(pose_dir = "/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuchVideoICP/ob_in_cam")
        self.trans_movement_2 = ResultPlotter.calcTransMovement(pose_dir = "/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuchVideoICP/ob_in_cam")

    def plotResults(self):
        #x = range(0,len(y))
        #plt.hist(a)
        ax = plt.gca()
        ax.set_ylim([0, 1])
        #plt.plot(x_masked,add_pvnet_orig, "-m", label ="ADD PVNet orig")
        # plt.plot(x,confidence_kpt_0, label ="Confidences kpt 0")
        # plt.plot(x,confidence_kpt_1, label ="Confidences kpt 1")
        # plt.plot(x,confidence_kpt_2, label ="Confidences kpt 2")
        # plt.plot(x,confidence_kpt_3, label ="Confidences kpt 3")
        # plt.plot(x,confidence_kpt_4, label ="Confidences kpt 4")
        # plt.plot(x,confidence_kpt_5, label ="Confidences kpt 5")
        # plt.plot(x,confidence_kpt_6, label ="Confidences kpt 6")
        # plt.plot(x,confidence_kpt_7, label ="Confidences kpt 7")
        # plt.plot(x,confidence_kpt_8, label ="Confidences kpt 8")
        
        #plt.plot(x,avg, label ="avg")
        #plt.plot(x,stabw, label ="stabw")

        #plt.plot(x,add_pvnet_upnp, "-r",label ="ADD PVNet upnp")
        plt.plot(self.x, self.add_bundle_orig, label="ADD BundleSDF original")
        #plt.plot(x, rot_movement_2, label="Rot movement")
        #plt.plot(self.x_masked, self.rot_movement_2, label="Rot movement")
        #plt.plot(self.x_masked, self.trans_movement_2, label="Trans movement")
        #plt.plot(x, add_bundle_periodic_orig, label="ADD BundleSDF periodic orig")
        plt.plot(self.x, self.add_bundle_limit_rot, label="ADD BundleSDF Limit Trans Rot")
        #plt.plot(self.x, self.add_bundle_icp, label="ADD BundleSDF ICP")
        plt.plot(self.x_masked, self.add_bundle_icp_masked, label="ADD BundleSDF ICP Masked")
        plt.plot(self.x, self.add_bundle_occ_aware, label="ADD BundleSDF Occlusion aware")
        #plt.plot(x, add_bundle_limit_rot, label="ADD limit rot")
        #plt.plot(x, add_bundle_periodic_upnp, label="ADD BundleSDF periodic upnp")
        plt.legend(loc="upper left")
        plt.show()

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
        return poses
    
    @staticmethod
    def calcRotMovement(pose_dir):
        poses = ResultPlotter.loadPoses(pose_dir)
        rot_movements = [0]
        for idx,pose in enumerate(poses):
            if idx > 0:
                if pose.round(decimals=6)[2,3] < 0.001 :
                    continue
                old_rot_idx = idx - 1
                while (poses[old_rot_idx].round(decimals=6)[2,3] < 0.001):
                    old_rot_idx -= 1
                old_rot = poses[old_rot_idx][:3,:3]
                rot_mat = pose[:3,:3]
                rot_movement = np.sum(np.abs(old_rot - rot_mat))
                rot_movements.append(rot_movement)

        return np.array(rot_movements) 

    @staticmethod
    def calcTransMovement(pose_dir):
        poses = ResultPlotter.loadPoses(pose_dir)
        trans_movements = [0]
        for idx,pose in enumerate(poses):
            if idx > 0:
                if pose.round(decimals=6)[2,3] < 0.001 :
                    continue
                old_t_vec_idx = idx - 1
                while (poses[old_t_vec_idx].round(decimals=6)[2,3] < 0.001):
                    old_t_vec_idx -= 1
                old_t_vec = poses[old_t_vec_idx][:3,3]
                t_vec = pose[:3,3]
                trans_movement = dist(old_t_vec, t_vec)
                trans_movements.append(trans_movement)

        return np.array(trans_movements)
    
    #calculates a mask array for valid poses only
    @staticmethod
    def calcMask(pose_dir):
        mask = []
        poses = ResultPlotter.loadPoses(pose_dir)
        for idx,pose in enumerate(poses):
            if pose.round(decimals=6)[2,3] < 0.001:
                mask.append(False)
            else:
                mask.append(True)
        return np.array(mask)



if __name__ == "__main__":
    result_plot = ResultPlotter()
    result_plot.plotResults()
