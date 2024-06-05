import numpy as np
import matplotlib.pyplot as plt
import os

import time

def calcRotMovement():

    pose_dir = "/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuchVideoPeriodicPVNet/ob_in_cam"
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
        #print(pose.round(decimals=6))

        if idx > 0:
            old_rot = poses[idx - 1][:3,:3]
            rot_mat = pose[:3,:3]
            rot_movement = np.sum(np.abs(old_rot - rot_mat))
            rot_movements.append(rot_movement)

    return np.array(rot_movements) 

if __name__ == "__main__":



    load_arr = np.load("/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/benchmarks/BuchVideo/ADD_PVNet_orig.npy", allow_pickle=True).item()
    add_pvnet_orig = load_arr["result_y"]
    mask = []
    pose_dir = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/outPVNet239/pose"
    pose_paths = os.listdir(pose_dir)
    pose_paths.sort()
    for idx,pose_file in enumerate(pose_paths):
        pose = None
        if pose_file.endswith(".txt"):
            pose = np.loadtxt(os.path.join(pose_dir, pose_file))
        elif pose_file.endswith(".npy"):
            pose = np.load(os.path.join(pose_dir, pose_file))
        else:
            continue
        #print(pose.round(decimals=6))
        if pose.round(decimals=6)[2,3] < 0.001:
            mask.append(False)
        else:
            mask.append(True)

    mask = np.array(mask)
    add_pvnet_orig = add_pvnet_orig[mask]
    load_arr = np.load("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/outPVNet239_temp/confidences_indiv.npy", allow_pickle=True).item()
    cov_invs = load_arr["result_y"] 
    confidence_sum = np.sum(np.abs(cov_invs), axis=2)
    confidence_sum = np.sum(confidence_sum, axis=2)
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
    confidence_kpt_0 = confidence_kpt_0[mask]
    confidence_kpt_1 = confidence_kpt_1[mask]
    confidence_kpt_2 = confidence_kpt_2[mask]
    confidence_kpt_3 = confidence_kpt_3[mask]
    confidence_kpt_4 = confidence_kpt_4[mask]
    confidence_kpt_5 = confidence_kpt_5[mask]
    confidence_kpt_6 = confidence_kpt_6[mask]
    confidence_kpt_7 = confidence_kpt_7[mask]
    confidence_kpt_8 = confidence_kpt_8[mask]

    confidence_sum_no_last = confidence_sum[:,:-1]
    stabw = np.std(confidence_sum_no_last,axis = 1)[mask]
    #stabw = np.sqrt(stabw)
    avg = np.average(confidence_sum_no_last,axis = 1)[mask]



    load_arr = np.load("/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/benchmarks/BuchVideo/ADD_PVNet_upnp.npy", allow_pickle=True).item()
    add_pvnet_upnp = load_arr["result_y"]
    add_pvnet_upnp = add_pvnet_upnp[mask]

    load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_orig.npy", allow_pickle=True).item()
    add_bundle_orig = load_arr["result_y"]

    load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_PeriodicPVNet_orig.npy", allow_pickle=True).item()
    add_bundle_periodic_orig = load_arr["result_y"]
    load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_PeriodicPVNet_upnp.npy", allow_pickle=True).item()
    add_bundle_periodic_upnp = load_arr["result_y"]


    rot_movement = np.load("outBuchVideoRotMovement/rot_movement/1699.npy", allow_pickle=True)
    trans_movement = np.load("outBuchVideoRotMovement/trans_movement/1699.npy", allow_pickle=True)
    
    rot_movement_2 = calcRotMovement()


    x = load_arr["ids"]
    x_masked = x[mask]

    
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
    #plt.plot(x, add_bundle_orig, label="ADD BundleSDF original")
    plt.plot(x, rot_movement, label="Rot movement")
    plt.plot(x, rot_movement_2, label="Rot 2 movement")
    plt.plot(x, trans_movement, label="Trans movement")
    plt.plot(x, add_bundle_periodic_orig, label="ADD BundleSDF periodic orig")
    #plt.plot(x, add_bundle_periodic_upnp, label="ADD BundleSDF periodic upnp")
    plt.legend(loc="upper left")
    plt.show()
