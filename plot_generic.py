import numpy as np
import matplotlib.pyplot as plt
import os

import time

if __name__ == "__main__":

    load_arr = np.load("/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/benchmarks/BuchVideo/ADD_PVNet_orig.npy", allow_pickle=True).item()
    add_1 = load_arr["result_y"]
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
    add_1 = add_1[mask]
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
    add_2 = load_arr["result_y"]
    add_2 = add_2[mask]

    x = load_arr["ids"]
    x = x[mask]

    
    #x = range(0,len(y))
    #plt.hist(a)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.plot(x,add_1, "-m", label ="ADD PVNet orig")
    # plt.plot(x,confidence_kpt_0, label ="Confidences kpt 0")
    # plt.plot(x,confidence_kpt_1, label ="Confidences kpt 1")
    # plt.plot(x,confidence_kpt_2, label ="Confidences kpt 2")
    # plt.plot(x,confidence_kpt_3, label ="Confidences kpt 3")
    # plt.plot(x,confidence_kpt_4, label ="Confidences kpt 4")
    # plt.plot(x,confidence_kpt_5, label ="Confidences kpt 5")
    # plt.plot(x,confidence_kpt_6, label ="Confidences kpt 6")
    # plt.plot(x,confidence_kpt_7, label ="Confidences kpt 7")
    # plt.plot(x,confidence_kpt_8, label ="Confidences kpt 8")
    
    plt.plot(x,avg, label ="avg")
    plt.plot(x,stabw, label ="stabw")

    #plt.plot(x,add_2, "-r",label ="ADD PVNet upnp")
    plt.legend(loc="upper left")
    plt.show()
