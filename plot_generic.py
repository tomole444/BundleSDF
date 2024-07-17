import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
from math import dist
import cv2
import time
import hashlib
import scienceplots

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from PyPDF2 import PdfReader, PdfWriter

class ResultPlotter:
    graph1 = None 
    graph2 = None 
    x = None 
    y1 = None 
    y2 = None 
    def __init__(self):
        self.loadADDResults()
        self.loadMaskResults()


    def loadADDResults(self):
        load_arr = np.load("/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/benchmarks/BuchVideo/ADD_PVNet_orig.npy", allow_pickle=True).item()
        self.add_pvnet_orig = load_arr["result_y"]

        pose_dir = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/outPVNet239/pose"
        self.mask = ResultPlotter.calcMask(pose_dir=pose_dir)


        
        self.x = load_arr["ids"]


        #self.add_pvnet_orig = self.add_pvnet_orig[self.mask]
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
        
        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_NoNerf.npy", allow_pickle=True).item()
        self.add_bundle_nonerf = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_NoNerfPVNet.npy", allow_pickle=True).item()
        self.add_bundle_nonerf_pvnet = load_arr["result_y"]
        
        load_arr = np.load("benchmarks/BuchVideo/ADD_Test.npy", allow_pickle=True).item()
        self.add_test = load_arr["result_y"]

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


        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_Occlusion_Aware.npy", allow_pickle=True).item()
        self.add_bundle_occ_aware = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_Occlusion_Aware_check_limit.npy", allow_pickle=True).item()
        self.add_bundle_occ_aware_check_limit = load_arr["result_y"]
        self.add_bundle_occ_aware_check_limit_trans_err = load_arr["trans_err"]
        self.add_bundle_occ_aware_check_limit_rot_err = load_arr["rot_err"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_Occlusion_Aware_force_pvnet.npy", allow_pickle=True).item()
        self.add_bundle_occ_aware_force_pvnet = load_arr["result_y"]


        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_feature_matching_spike.npy", allow_pickle=True).item()
        self.add_bundle_feature_matching_spike = load_arr["result_y"]
        

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_pose_regression_2.npy", allow_pickle=True).item()
        self.add_bundle_pose_regression = load_arr["result_y"]
        
        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_cutie_first_offline_segmentation.npy", allow_pickle=True).item()
        self.add_bundle_cutie_first_offline_segmentation = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_orig_cutie_segmentation.npy", allow_pickle=True).item()
        self.add_bundle_orig_cutie_segmentation = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_orig_xmem_segmentation.npy", allow_pickle=True).item()
        self.add_bundle_orig_xmem_segmentation = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_first_pvnet_cutie_segmentation.npy", allow_pickle=True).item()
        self.add_bundle_first_pvnet_cutie_segmentation = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_pvnet_segmentation_only.npy", allow_pickle=True).item()
        self.add_bundle_pvnet_seg_only = load_arr["result_y"]



        #BuchVideo2


        load_arr = np.load("benchmarks/BuchVideo2/ADD_BundleSDF_orig.npy", allow_pickle=True).item()
        self.add_bundle_orig_buch_2 = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo2/ADD_BundleSDF_pose_regression_2.npy", allow_pickle=True).item()
        self.add_bundle_pose_regression_buch_2 = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo2/ADD_PVNet_upnp_Big_dataset.npy", allow_pickle=True).item()
        self.add_pvnet_orig_buch_2 = load_arr["result_y"]
        self.mask = ResultPlotter.calcMask(pose_dir="/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo2/outPVNet239upnp/pose")
        self.x_masked = self.x[self.mask]

        load_arr = np.load("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo2/outPVNet239upnp/confidences_indiv.npy", allow_pickle=True).item()
        cov_invs = load_arr["result_y"] 

        self.confidence_sum = np.sum(np.abs(cov_invs), axis=2)
        #confidence_sum = np.sum(confidence_sum, axis=2)
        self.confidence_sum = 1-(self.confidence_sum / 5)
        self.confidence_kpt_0 = self.confidence_sum[:,0] #1 - (confidence_sum[:,0]/ 5)
        self.confidence_kpt_1 = self.confidence_sum[:,1] #1 - (confidence_sum[:,1]/ 5)
        self.confidence_kpt_2 = self.confidence_sum[:,2] #1 - (confidence_sum[:,2]/ 5)
        self.confidence_kpt_3 = self.confidence_sum[:,3] #1 - (confidence_sum[:,3]/ 5)
        self.confidence_kpt_4 = self.confidence_sum[:,4] #1 - (confidence_sum[:,4]/ 5)
        self.confidence_kpt_5 = self.confidence_sum[:,5] #1 - (confidence_sum[:,5]/ 5)
        self.confidence_kpt_6 = self.confidence_sum[:,6] #1 - (confidence_sum[:,6]/ 5)
        self.confidence_kpt_7 = self.confidence_sum[:,7] #1 - (confidence_sum[:,7]/ 5)
        self.confidence_kpt_8 = self.confidence_sum[:,8] #1 - (confidence_sum[:,8]/ 5)
        self.confidence_kpt_0 = np.append(self.confidence_kpt_0,0)
        self.confidence_kpt_1 = np.append(self.confidence_kpt_1,0)
        self.confidence_kpt_2 = np.append(self.confidence_kpt_2,0)
        self.confidence_kpt_3 = np.append(self.confidence_kpt_3,0)
        self.confidence_kpt_4 = np.append(self.confidence_kpt_4,0)
        self.confidence_kpt_5 = np.append(self.confidence_kpt_5,0)
        self.confidence_kpt_6 = np.append(self.confidence_kpt_6,0)
        self.confidence_kpt_7 = np.append(self.confidence_kpt_7,0)
        self.confidence_kpt_8 = np.append(self.confidence_kpt_8,0)
        self.confidence_kpt_0 = self.confidence_kpt_0[self.mask]
        self.confidence_kpt_1 = self.confidence_kpt_1[self.mask]
        self.confidence_kpt_2 = self.confidence_kpt_2[self.mask]
        self.confidence_kpt_3 = self.confidence_kpt_3[self.mask]
        self.confidence_kpt_4 = self.confidence_kpt_4[self.mask]
        self.confidence_kpt_5 = self.confidence_kpt_5[self.mask]
        self.confidence_kpt_6 = self.confidence_kpt_6[self.mask]
        self.confidence_kpt_7 = self.confidence_kpt_7[self.mask]
        self.confidence_kpt_8 = self.confidence_kpt_8[self.mask]

        confidence_sum_no_last = confidence_sum[:,:-1]
        self.stabw = np.std(confidence_sum_no_last,axis = 1)[self.mask]
        #stabw = np.sqrt(stabw)
        self.avg = np.average(confidence_sum_no_last,axis = 1)[self.mask]

        self.err_detections = np.where(self.mask, 0, 0.8)



        #self.mask_count = ResultPlotter.countVisablePixels("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/masks")
        #np.save("benchmarks/BuchVideo/mask_visib_pixels.npy", self.mask_count)
        self.mask_count = np.load("benchmarks/BuchVideo/mask_visib_pixels.npy")

        #rot_movement = np.load("outBuchVideoNoLimiting/rot_movement/1699.npy", allow_pickle=True)

        #trans_movement = np.load("outBuchVideoNoLimiting/trans_movement/1699.npy", allow_pickle=True)
        
        self.rot_movement_2 = ResultPlotter.calcRotMovement(pose_dir = "/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuchVideoICP/ob_in_cam")
        self.trans_movement_2 = ResultPlotter.calcTransMovement(pose_dir = "/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuchVideoICP/ob_in_cam")
    
    def loadMaskResults(self):
        load_arr = np.load("benchmarks/BuchVideo/mask_analysis/Metrics_pvnet.npy", allow_pickle=True).item()
        self.iou_pvnet = load_arr["iou"]

        load_arr = np.load("benchmarks/BuchVideo/mask_analysis/Metrics_first_mask_pvnet_cutie.npy", allow_pickle=True).item()
        self.iou_first_mask_pvnet_cutie = load_arr["iou"]

        load_arr = np.load("benchmarks/BuchVideo/mask_analysis/Metrics_first_mask_pvnet_xmem.npy", allow_pickle=True).item()
        self.iou_first_mask_pvnet_xmem = load_arr["iou"]

    
    def plotMaskResults(self):
        plt.switch_backend('TkAgg')
        #plt.rc ('font', size = 30)
        fig =plt.figure(figsize=(16, 9), dpi=(1920/16))
        ax = plt.gca()
        #ax.set_ylim([0, 1])
        ax.set_xlim([0, len(self.x)])

        plt.plot(self.x, self.iou_pvnet, label = "IOU Pure PVNet")
        plt.plot(self.x, self.iou_first_mask_pvnet_cutie, label = "IOU First PVNet Cutie")
        plt.plot(self.x, self.iou_first_mask_pvnet_xmem, label = "IOU First PVNet XMem")

        plt.legend(loc="upper right")
        ax.set_xlabel("Frame")
        ax.set_ylabel("IOU")
        ax.grid(True)
        
        ax.set_title('IOU comparison', fontsize = 40, fontweight ='bold')
        plt.show()

    def plotADDResults(self, use_tk_backend = True):
        #x = range(0,len(y))
        #plt.hist(a)
        #matplotlib                3.7.1
        #matplotlib-inline         0.1.7

        if use_tk_backend:
            plt.switch_backend('TkAgg')
        #plt.style.use(['science','ieee'])

        plt.rc ('font', size = 30)
        fig =plt.figure(figsize=(16, 9), dpi=(1920/16))
        ax = plt.gca()
        ax.set_ylim([0, 1])
        ax.set_xlim([0, len(self.x)])

        ResultPlotter.x = self.x
        ResultPlotter.y1 = self.add_bundle_orig
        ResultPlotter.y2 = self.add_bundle_feature_matching_spike

        #ResultPlotter.graph1, = ax.plot([0], [0], label="BundleSDF original")
        #ResultPlotter.graph2, = ax.plot([0], [0], label = "Current Implementation")

        #plt.plot(self.x,self.add_pvnet_orig, ResultPlotter.string_to_rgb("ADD PVNet orig"), label ="ADD PVNet orig")
        # plt.plot(self.x_masked, self.confidence_kpt_0, label ="Confidences kpt 0")
        # plt.plot(self.x_masked, self.confidence_kpt_1, label ="Confidences kpt 1")
        # plt.plot(self.x_masked, self.confidence_kpt_2, label ="Confidences kpt 2")
        # plt.plot(self.x_masked, self.confidence_kpt_3, label ="Confidences kpt 3")
        # plt.plot(self.x_masked, self.confidence_kpt_4, label ="Confidences kpt 4")
        # plt.plot(self.x_masked, self.confidence_kpt_5, label ="Confidences kpt 5")
        # plt.plot(self.x_masked, self.confidence_kpt_6, label ="Confidences kpt 6")
        # plt.plot(self.x_masked, self.confidence_kpt_7, label ="Confidences kpt 7")
        # plt.plot(self.x_masked, self.confidence_kpt_8, label ="Confidences kpt 8")
        
        #plt.plot(self.x_masked, self.avg, label ="avg")
        #plt.plot(self.x_masked, self.stabw, label ="stabw")


        #plt.plot(x,add_pvnet_upnp, "-r",label ="ADD PVNet upnp")
        #plt.plot(self.x, self.add_bundle_orig, label="Original")
        plt.plot(self.x, self.add_bundle_nonerf, label="No NeRF")
        plt.plot(self.x, self.add_bundle_nonerf_pvnet, label="First Estimation PVNet")
        #plt.plot(x, rot_movement_2, label="Rot movement")
        #plt.plot(self.x_masked, self.rot_movement_2, label="Rot movement")
        #plt.plot(self.x_masked, self.trans_movement_2, label="Trans movement")
        #plt.plot(x, add_bundle_periodic_orig, label="ADD BundleSDF periodic orig")
        #plt.plot(self.x, self.add_bundle_limit_rot_trans, label="ADD BundleSDF Limit Trans Rot")
        #plt.plot(self.x, self.add_bundle_icp, label="ADD BundleSDF ICP")
        #plt.plot(self.x, self.add_bundle_occ_aware_check_limit, label="ADD BundleSDF Occlusion aware check limits") #1380 problematic -> full occlusion
        #plt.plot(self.x, self.add_bundle_occ_aware_check_limit_trans_err, label="ADD BundleSDF Occlusion aware trans err") 
        #plt.plot(self.x, self.add_bundle_occ_aware_check_limit_rot_err, label="ADD BundleSDF Occlusion aware rot err")
        #plt.plot(self.x, self.add_bundle_occ_aware_force_pvnet, label="ADD BundleSDF Occlusion aware force pvnet") #1380 problematic -> full occlusion
        #plt.plot(self.x,self.add_bundle_feature_matching_spike, label = "ADD feature spike prevention")
        #plt.plot(self.x,self.add_bundle_pose_regression, label = "ADD Pose regression")
        #plt.plot(self.x,self.add_bundle_pose_regression, label = "ADD Pose regression 2")
        #plt.plot(self.x,self.add_bundle_cutie_first_offline_segmentation, label = "ADD First Offline Cutie segmentation")
        #plt.plot(self.x,self.add_bundle_orig_cutie_segmentation, label = "ADD Orig Cutie segmentation")
        #plt.plot(self.x,self.add_bundle_orig_xmem_segmentation, label = "ADD Orig XMem segmentation")
        #plt.plot(self.x,self.add_bundle_first_pvnet_cutie_segmentation, label = "ADD First PVNet Cutie Segmentation")
        #plt.plot(self.x,self.add_bundle_pvnet_seg_only, label = "ADD PVNet Only Segmentation")
        #plt.plot(self.x,self.add_test, label = "ADD First PVNet Cutie Segmentation_2")




        # plt.plot(self.x,self.add_bundle_orig_buch_2, label = "ADD BundleSDF orig BuchVideo2")
        # plt.plot(self.x,self.add_bundle_pose_regression_buch_2, label = "ADD BundleSDF pose_regression BuchVideo2")
        # plt.plot(self.x,self.add_pvnet_orig_buch_2, label = "ADD PVNet BuchVideo2")

        #jumps = ResultPlotter.getJumps(self.add_bundle_occ_aware_masked,self.x_masked)
        #print("jumps at", jumps)
        #plt.plot(self.x_masked, self.add_bundle_occ_aware_masked, label="ADD BundleSDF Occlusion aware masked")

        #plt.plot(self.x, self.err_detections, "-r", label = "No Detections")
        #634 problematic

        #plt.plot(self.x, self.mask_count, label = "Mask visib Pixels")

        #plt.plot(x, add_bundle_limit_rot, label="ADD limit rot")
        #plt.plot(x, add_bundle_periodic_upnp, label="ADD BundleSDF periodic upnp")


        plt.legend(loc="upper right")
        ax.set_xlabel("Frame-ID")
        ax.set_ylabel("ADD [m]")
        ax.grid(True)
        
        #ax.set_title('ADD comparison', fontsize = 40, fontweight ='bold')

        #fps = 25

        #ani = FuncAnimation(fig, self.animate, frames=len(self.x), interval=int(1/fps * 1e3))
        #writer = FFMpegWriter(fps=fps, metadata=dict(artist='Tom Leyh'), extra_args=['-vcodec', 'libx264'])
        #ani.save('/home/thws_robotik/Downloads/ADD_own_implementation.mp4', writer=writer)
        #plt.show()
    
    def exportPlot(self, path:str, white_border:bool = False):
        is_pdf = False
        if(path.endswith(".pdf")):
            is_pdf = True
            path = path.replace(".pdf", ".svg")
        if white_border:
            plt.savefig(path)
        else:
            plt.savefig(path, bbox_inches='tight', transparent=True)

        if is_pdf:
            dir = os.path.dirname(path)
            file_name = os.path.basename(path)
            drawing = svg2rlg(path)
            
            # Render the drawing to PDF
            pdf_filename = os.path.join(dir, file_name.replace(".svg", ".pdf"))
            renderPDF.drawToFile(drawing, pdf_filename)

    @staticmethod
    def animate(frame):
        ResultPlotter.graph1.set_xdata(ResultPlotter.x[:frame])
        ResultPlotter.graph2.set_xdata(ResultPlotter.x[:frame])
        ResultPlotter.graph1.set_ydata(ResultPlotter.y1[:frame])
        ResultPlotter.graph2.set_ydata(ResultPlotter.y2[:frame])

        return ResultPlotter.graph1, ResultPlotter.graph2

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
    
    def countVisablePixels(mask_dir):
        mask_paths = os.listdir(mask_dir)
        mask_paths.sort()
        mask_count = []
        for idx,mask_file in enumerate(mask_paths):
            print("loading ", mask_file)
            mask = cv2.imread(os.path.join(mask_dir,mask_file), cv2.IMREAD_GRAYSCALE)
            mask_count.append(len(mask[mask > 1]))

        return np.array(mask_count)
    @staticmethod
    def getJumps(masked_add_arr, x_masked, threshold = 0.2):
        last_avg = 0
        last_four = np.array([0,0,0,0])
        jumps = []
        for idx, add_metric in enumerate(masked_add_arr):
            if(np.abs(add_metric - last_avg) > threshold):
                jumps.append(x_masked[idx])
            last_four = np.roll(last_four,1)
            last_four[0] = add_metric
            last_avg = np.average(last_four)
        return np.array(jumps) 

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

    @staticmethod
    def string_to_rgb(input_string):
        hash_object = hashlib.md5(input_string.encode())
        hex_dig = hash_object.hexdigest()
        
        r = int(hex_dig[0:2], 16)
        g = int(hex_dig[2:4], 16)
        b = int(hex_dig[4:6], 16)
        
        return (r, g, b)


if __name__ == "__main__":
    result_plot = ResultPlotter()
    result_plot.plotADDResults()
    result_plot.exportPlot("plots/ADD_BundleSDF_no_nerf_pvnet_2.pdf")
    #result_plot.plotMaskResults()
