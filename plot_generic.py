import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
from math import dist
import cv2
import time
import hashlib
import scienceplots
from TimeAnalyser import TimeAnalyser

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

        self.time_keeper = None
        self.loadTimingResults()

        self.setupPlot()


    def loadADDResults(self):
        diameter = 0.211
        ADD_logpath = "plots/BuchVideo/ADD/ADD-whole.txt"
        full_text_add_combined = ""
        latex_add_combined = ""


        load_arr = np.load("/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/benchmarks/BuchVideo/ADD_PVNet_orig.npy", allow_pickle=True).item()
        self.add_pvnet_orig = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_PVNet_orig: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_PVNet_orig & " + latex_add

        pose_dir = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/outPVNet239/pose"

        self.x = load_arr["ids"]
        self.mask = ResultPlotter.calcMask(pose_dir=pose_dir)
        self.x_masked = self.x[self.mask]
        #self.add_pvnet_orig[np.invert( self.mask)] = -1


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
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_PVNet_upnp: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_PVNet_upnp & " + latex_add
        
        self.mask_upnp = ResultPlotter.calcMask(pose_dir="/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/outPVNet239_upnp/pose")
        self.x_masked_upnp = self.x[self.mask_upnp]
        
        #self.add_pvnet_upnp[np.invert(self.mask_upnp)] = -1

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_orig.npy", allow_pickle=True).item()
        self.add_bundle_orig = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_orig: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_orig & " + latex_add
        
        
        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_NoNerf.npy", allow_pickle=True).item()
        self.add_bundle_nonerf = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_NoNerf: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_NoNerf & " + latex_add

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_NoNerfPVNet_2.npy", allow_pickle=True).item()
        self.add_bundle_nonerf_pvnet = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_NoNerfPVNet_2: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_NoNerfPVNet_2 & " + latex_add
        
        load_arr = np.load("benchmarks/BuchVideo/ADD_Test.npy", allow_pickle=True).item()
        self.add_test = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_PeriodicPVNet_orig.npy", allow_pickle=True).item()
        self.add_bundle_periodic_orig = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_PeriodicPVNet_orig: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_PeriodicPVNet_orig & " + latex_add
        
        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_PeriodicPVNet_upnp.npy", allow_pickle=True).item()
        self.add_bundle_periodic_upnp = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_PeriodicPVNet_upnp: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_PeriodicPVNet_upnp & " + latex_add

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_LimitRot.npy", allow_pickle=True).item()
        self.add_bundle_limit_rot = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_LimitRot: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_LimitRot & " + latex_add

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_LimitRotTrans.npy", allow_pickle=True).item()
        self.add_bundle_limit_rot_trans = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_LimitRotTrans: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_LimitRotTrans & " + latex_add

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_ICP.npy", allow_pickle=True).item()
        self.add_bundle_icp = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_ICP: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_ICP & " + latex_add


        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_Occlusion_Aware.npy", allow_pickle=True).item()
        self.add_bundle_occ_aware = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_Occlusion_Aware: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_Occlusion_Aware & " + latex_add

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_Occlusion_Aware_check_limit.npy", allow_pickle=True).item()
        self.add_bundle_occ_aware_check_limit = load_arr["result_y"]
        self.add_bundle_occ_aware_check_limit_trans_err = load_arr["trans_err"]
        self.add_bundle_occ_aware_check_limit_rot_err = load_arr["rot_err"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_Occlusion_Aware_check_limit: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_Occlusion_Aware_check_limit & " + latex_add

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_Occlusion_Aware_force_pvnet.npy", allow_pickle=True).item()
        self.add_bundle_occ_aware_force_pvnet = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_Occlusion_Aware_force_pvnet: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_Occlusion_Aware_force_pvnet & " + latex_add


        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_feature_matching_spike.npy", allow_pickle=True).item()
        self.add_bundle_feature_matching_spike = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_feature_matching_spike: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_feature_matching_spike & " + latex_add
        

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_pose_regression_2.npy", allow_pickle=True).item()
        self.add_bundle_pose_regression_2 = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_pose_regression_2: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_pose_regression_2 & " + latex_add

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_pose_regression_-4.npy", allow_pickle=True).item()
        self.add_bundle_pose_regression_minus_4 = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_pose_regression_-4: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_pose_regression_-4 & " + latex_add
        
        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_cutie_first_offline_segmentation.npy", allow_pickle=True).item()
        self.add_bundle_cutie_first_offline_segmentation = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_cutie_first_offline_segmentation: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_cutie_first_offline_segmentation & " + latex_add

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_orig_cutie_segmentation.npy", allow_pickle=True).item()
        self.add_bundle_orig_cutie_segmentation = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_orig_cutie_segmentation: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_orig_cutie_segmentation & " + latex_add

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_orig_xmem_segmentation.npy", allow_pickle=True).item()
        self.add_bundle_orig_xmem_segmentation = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_orig_xmem_segmentation: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_orig_xmem_segmentation & " + latex_add

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_first_pvnet_cutie_segmentation.npy", allow_pickle=True).item()
        self.add_bundle_first_pvnet_cutie_segmentation = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_first_pvnet_cutie_segmentation: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_first_pvnet_cutie_segmentation & " + latex_add

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_pvnet_segmentation_only.npy", allow_pickle=True).item()
        self.add_bundle_pvnet_seg_only = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], diameter)
        full_text_add_combined += "ADD_BundleSDF_pvnet_segmentation_only: " + full_text_add + "\n\n"
        latex_add_combined += "ADD_BundleSDF_pvnet_segmentation_only & " + latex_add

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_current_implementation.npy", allow_pickle=True).item()
        self.add_bundle_current_implementation = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_union_occlusion.npy", allow_pickle=True).item()
        self.add_bundle_union_occlusion = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_extrapolated_poses_only_2.npy", allow_pickle=True).item()
        self.add_bundle_extrapolated_poses_only_2 = load_arr["result_y"]
        self.x_extrapolated_poses_only_2 = load_arr["ids"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_extrapolated_poses_only_-4.npy", allow_pickle=True).item()
        self.add_bundle_extrapolated_poses_only_minus_4 = load_arr["result_y"]
        self.x_extrapolated_poses_only_minus_4 = load_arr["ids"]





        #BuchVideo2


        load_arr = np.load("benchmarks/BuchVideo2/ADD_BundleSDF_orig.npy", allow_pickle=True).item()
        self.add_bundle_orig_buch_2 = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo2/ADD_BundleSDF_pose_regression_2.npy", allow_pickle=True).item()
        self.add_bundle_pose_regression_buch_2 = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo2/ADD_PVNet_upnp_Big_dataset.npy", allow_pickle=True).item()
        self.add_pvnet_orig_buch_2 = load_arr["result_y"]


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
        self.confidence_kpt_0 = self.confidence_kpt_0[self.mask_upnp]
        self.confidence_kpt_1 = self.confidence_kpt_1[self.mask_upnp]
        self.confidence_kpt_2 = self.confidence_kpt_2[self.mask_upnp]
        self.confidence_kpt_3 = self.confidence_kpt_3[self.mask_upnp]
        self.confidence_kpt_4 = self.confidence_kpt_4[self.mask_upnp]
        self.confidence_kpt_5 = self.confidence_kpt_5[self.mask_upnp]
        self.confidence_kpt_6 = self.confidence_kpt_6[self.mask_upnp]
        self.confidence_kpt_7 = self.confidence_kpt_7[self.mask_upnp]
        self.confidence_kpt_8 = self.confidence_kpt_8[self.mask_upnp]

        confidence_sum_no_last = confidence_sum[:,:-1]
        self.stabw = np.std(confidence_sum_no_last,axis = 1)[self.mask_upnp]
        #stabw = np.sqrt(stabw)
        self.avg = np.average(confidence_sum_no_last,axis = 1)[self.mask_upnp]

        self.err_detections = np.where(self.mask, 0, 0.8)



        #self.mask_count = ResultPlotter.countVisablePixels("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/masks")
        #np.save("benchmarks/BuchVideo/mask_visib_pixels.npy", self.mask_count)
        self.mask_count = np.load("benchmarks/BuchVideo/mask_visib_pixels.npy")

        #rot_movement = np.load("outBuchVideoNoLimiting/rot_movement/1699.npy", allow_pickle=True)

        #trans_movement = np.load("outBuchVideoNoLimiting/trans_movement/1699.npy", allow_pickle=True)
        
        self.rot_movement_2 = ResultPlotter.calcRotMovement(pose_dir = "/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuchVideoICP/ob_in_cam")
        self.trans_movement_2 = ResultPlotter.calcTransMovement(pose_dir = "/home/thws_robotik/Documents/Leyh/6dpose/detection/BundleSDF/outBuchVideoICP/ob_in_cam")

        with open(ADD_logpath, 'w') as datei:
            datei.write(full_text_add_combined)
            datei.write("\n\n\n\n")
            datei.write(latex_add_combined)
    
    def loadMaskResults(self):
        load_arr = np.load("benchmarks/BuchVideo/mask_analysis/Metrics_pvnet.npy", allow_pickle=True).item()
        self.iou_pvnet = load_arr["iou"]
        self.pixel_acc_pvnet = load_arr["pixel_acc"]
        self.precision_acc_pvnet = load_arr["precision"]
        self.recall_pvnet = load_arr["recall"]
        self.dice_pvnet = load_arr["dice"]

        load_arr = np.load("benchmarks/BuchVideo/mask_analysis/Metrics_first_mask_pvnet_cutie.npy", allow_pickle=True).item()
        self.iou_first_mask_pvnet_cutie = load_arr["iou"]
        self.pixel_acc_first_mask_pvnet_cutie = load_arr["pixel_acc"]
        self.precision_acc_first_mask_pvnet_cutie = load_arr["precision"]
        self.recall_first_mask_pvnet_cutie = load_arr["recall"]
        self.dice_first_mask_pvnet_cutie = load_arr["dice"]

        load_arr = np.load("benchmarks/BuchVideo/mask_analysis/Metrics_first_mask_offline_cutie.npy", allow_pickle=True).item()
        self.iou_first_mask_offline_cutie = load_arr["iou"] 
        self.pixel_acc_first_mask_offline_cutie = load_arr["pixel_acc"]
        self.precision_acc_first_mask_offline_cutie = load_arr["precision"]
        self.recall_first_mask_offline_cutie = load_arr["recall"]
        self.dice_first_mask_offline_cutie = load_arr["dice"]       

        load_arr = np.load("benchmarks/BuchVideo/mask_analysis/Metrics_first_mask_pvnet_xmem.npy", allow_pickle=True).item()
        self.iou_first_mask_pvnet_xmem = load_arr["iou"]
        self.pixel_acc_first_mask_pvnet_xmem = load_arr["pixel_acc"]
        self.precision_acc_first_mask_pvnet_xmem = load_arr["precision"]
        self.recall_first_mask_pvnet_xmem = load_arr["recall"]
        self.dice_first_mask_pvnet_xmem = load_arr["dice"]

        load_arr = np.load("benchmarks/BuchVideo/mask_analysis/Metrics_first_mask_offline_xmem.npy", allow_pickle=True).item()
        self.iou_first_mask_offline_xmem = load_arr["iou"]
        self.pixel_acc_first_mask_offline_xmem = load_arr["pixel_acc"]
        self.precision_acc_first_mask_offline_xmem = load_arr["precision"]
        self.recall_first_mask_offline_xmem = load_arr["recall"]
        self.dice_first_mask_offline_xmem = load_arr["dice"]

    def loadTimingResults(self, timing_file_path = "benchmarks/BuchVideo/time_analysis/timing_pose_regression_2.npy"):
        self.time_keeper = TimeAnalyser()
        self.time_keeper.load(timing_file_path)
        timing_log_path = "plots/BuchVideo/timing/BundleSDF_pose_regression_2.txt"
        WRITE_LOG = False
        

        keys = self.time_keeper.time_save.keys()
        #print("Loaded timer with keys: ", keys)

        ANALYSE_ORIGINAL = False

        timing_log_str = f"Average times (ANALYSE_ORIGINAL = {ANALYSE_ORIGINAL}, timing_file_path = {timing_file_path}): " + "\n" 
        
        time_pair_preprocessing = ("preprocessing", "preprocessing_done")
        self.time_pair_preprocessing_ids , self.time_pair_preprocessing_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_preprocessing)
        timing_log_str += "time_pair_preprocessing_execution_times & " +  str(np.round(np.average(self.time_pair_preprocessing_execution_times), 5)) + "s " + "\\\\\n"
        
        time_pair_run = ("run", "run_done")

        #time_pair_process_new_frame = ("process_new_frame", "invalidatePixelsByMask")


        if ANALYSE_ORIGINAL:
            time_pair_invalidatePixelsByMask = ("invalidatePixelsByMask", "pointCloudDenoise")
            time_pair_pointCloudDenoise = ("pointCloudDenoise", "find_corres")
            time_pair_find_corres = ("find_corres", "len(matches)<min_match_with_ref")
            time_pair_min_match_with_ref = ("len(matches)<min_match_with_ref", "procrustesByCorrespondence")
            time_pair_procrustesByCorrespondence = ("procrustesByCorrespondence", "selectKeyFramesForBA")
            time_pair_selectKeyFramesForBA = ("selectKeyFramesForBA", "getFeatureMatchPairs")
            time_pair_getFeatureMatchPairs = ("getFeatureMatchPairs", "optimizeGPU")
            time_pair_optimizeGPU = ("optimizeGPU", "checkAndAddKeyframe")
            time_pair_checkAndAddKeyframe = ("checkAndAddKeyframe", "process_new_frame_done")
            time_pair_process_frame = ("process_new_frame", "process_new_frame_done")
            time_pair_nerf_start = ("nerf_start","nerf_end")
            time_pair_nerf_pose_adaptation = ("nerf_pose_adaptation","nerf_pose_adaptation_end")
            time_pair_rematch_after_nerf = ("rematch_after_nerf","rematch_after_nerf_end")

            self.time_pair_invalidatePixelsByMask_ids, self.time_pair_invalidatePixelsByMask_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_invalidatePixelsByMask)
            self.time_pair_pointCloudDenoise_ids, self.time_pair_pointCloudDenoise_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_pointCloudDenoise)
            self.time_pair_find_corres_ids, self.time_pair_find_corres_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_find_corres)
            self.time_pair_min_match_with_ref_ids, self.time_pair_min_match_with_ref_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_min_match_with_ref)
            self.time_pair_procrustesByCorrespondence_ids, self.time_pair_procrustesByCorrespondence_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_procrustesByCorrespondence)
            self.time_pair_selectKeyFramesForBA_ids, self.time_pair_selectKeyFramesForBA_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_selectKeyFramesForBA)
            self.time_pair_getFeatureMatchPairs_ids, self.time_pair_getFeatureMatchPairs_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_getFeatureMatchPairs)
            self.time_pair_optimizeGPU_ids, self.time_pair_optimizeGPU_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_optimizeGPU)
            self.time_pair_checkAndAddKeyframe_ids, self.time_pair_checkAndAddKeyframe_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_checkAndAddKeyframe)
            self.time_pair_process_frame_ids, self.time_pair_process_frame_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_process_frame)
            self.time_pair_nerf_start_ids, self.time_pair_nerf_start_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_nerf_start)
            self.time_pair_nerf_pose_adaptation_ids, self.time_pair_nerf_pose_adaptation_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_nerf_pose_adaptation)
            self.time_pair_rematch_after_nerf_ids, self.time_pair_rematch_after_nerf_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_rematch_after_nerf)

            timing_log_str += "time_pair_invalidatePixelsByMask_execution_times & " +  str(np.round(np.average(self.time_pair_invalidatePixelsByMask_execution_times), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_pointCloudDenoise_execution_times & " +  str(np.round(np.average(self.time_pair_pointCloudDenoise_execution_times), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_find_corres_execution_times & " +  str(np.round(np.average(self.time_pair_find_corres_execution_times), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_min_match_with_ref_execution_times & " +  str(np.round(np.average(self.time_pair_min_match_with_ref_execution_times), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_procrustesByCorrespondence_execution_times & " +  str(np.round(np.average(self.time_pair_procrustesByCorrespondence_execution_times), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_selectKeyFramesForBA_execution_times & " +  str(np.round(np.average(self.time_pair_selectKeyFramesForBA_execution_times), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_getFeatureMatchPairs_execution_times & " +  str(np.round(np.average(self.time_pair_getFeatureMatchPairs_execution_times), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_optimizeGPU_execution_times & " +  str(np.round(np.average(self.time_pair_optimizeGPU_execution_times), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_checkAndAddKeyframe_execution_times & " +  str(np.round(np.average(self.time_pair_checkAndAddKeyframe_execution_times), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_process_frame_execution_times & " +  str(np.round(np.average(self.time_pair_process_frame_execution_times), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_nerf_start_execution_times & " +  str(np.round(np.average(self.time_pair_nerf_start_execution_times), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_nerf_pose_adaptation_execution_times & " +  str(np.round(np.average(self.time_pair_nerf_pose_adaptation_execution_times), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_rematch_after_nerf_execution_times & " +  str(np.round(np.average(self.time_pair_rematch_after_nerf_execution_times), 5)) + "s " + "\\\\\n"

        else:
            time_pair_invalidatePixelsByMask = ("invalidatePixelsByMask","denoise_cloud")
            time_pair_denoise_cloud = ("denoise_cloud","pvnet_adjust_every")
            time_pair_find_corres_1 = ("find_corres_1","selectKeyFramesForBA")
            time_pair_selectKeyFramesForBA = ("selectKeyFramesForBA","getFeatureMatchPairs")
            time_pair_getFeatureMatchPairs = ("getFeatureMatchPairs","find_corres_2")
            time_pair_find_corres_2 = ("find_corres_2","optimizeGPU")
            time_pair_optimizeGPU = ("optimizeGPU","checkMovement_limits")
            time_pair_checkMovement_limits = ("checkMovement_limits","checkMovement_limits_end")
            time_pair_icp = ("icp","icp_end")
            time_pair_checkAndAddKeyframe = ("icp_end","process_new_frame_pvnet_done")
            time_pair_process_frame = ("process_new_frame_pvnet","process_new_frame_pvnet_done")

            self.time_pair_invalidatePixelsByMask_ids, self.time_pair_invalidatePixelsByMask_execution_time = self.time_keeper.getSyncTimeByFrameID(time_pair_invalidatePixelsByMask)
            self.time_pair_denoise_cloud_ids, self.time_pair_denoise_cloud_execution_time = self.time_keeper.getSyncTimeByFrameID(time_pair_denoise_cloud)
            self.time_pair_find_corres_1_ids, self.time_pair_find_corres_1_execution_time = self.time_keeper.getSyncTimeByFrameID(time_pair_find_corres_1)
            self.time_pair_selectKeyFramesForBA_ids, self.time_pair_selectKeyFramesForBA_execution_time = self.time_keeper.getSyncTimeByFrameID(time_pair_selectKeyFramesForBA)
            self.time_pair_getFeatureMatchPairs_ids, self.time_pair_getFeatureMatchPairs_execution_time = self.time_keeper.getSyncTimeByFrameID(time_pair_getFeatureMatchPairs)
            self.time_pair_find_corres_2_ids, self.time_pair_find_corres_2_execution_time = self.time_keeper.getSyncTimeByFrameID(time_pair_find_corres_2)
            self.time_pair_optimizeGPU_ids, self.time_pair_optimizeGPU_execution_time = self.time_keeper.getSyncTimeByFrameID(time_pair_optimizeGPU)
            self.time_pair_checkMovement_limits_ids, self.time_pair_checkMovement_limits_execution_time = self.time_keeper.getSyncTimeByFrameID(time_pair_checkMovement_limits)
            self.time_pair_icp_ids, self.time_pair_icp_execution_time = self.time_keeper.getSyncTimeByFrameID(time_pair_icp)
            self.time_pair_checkAndAddKeyframe_ids, self.time_pair_checkAndAddKeyframe_execution_time = self.time_keeper.getSyncTimeByFrameID(time_pair_checkAndAddKeyframe)
            self.time_pair_process_frame_ids, self.time_pair_process_frame_execution_time = self.time_keeper.getSyncTimeByFrameID(time_pair_process_frame)

            timing_log_str += "time_pair_invalidatePixelsByMask_execution_time & " + str(np.round(np.average(self.time_pair_invalidatePixelsByMask_execution_time), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_denoise_cloud_execution_time & " + str(np.round(np.average(self.time_pair_denoise_cloud_execution_time), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_find_corres_1_execution_time & " + str(np.round(np.average(self.time_pair_find_corres_1_execution_time), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_selectKeyFramesForBA_execution_time & " + str(np.round(np.average(self.time_pair_selectKeyFramesForBA_execution_time), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_getFeatureMatchPairs_execution_time & " + str(np.round(np.average(self.time_pair_getFeatureMatchPairs_execution_time), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_find_corres_2_execution_time & " + str(np.round(np.average(self.time_pair_find_corres_2_execution_time), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_optimizeGPU_execution_time & " + str(np.round(np.average(self.time_pair_optimizeGPU_execution_time), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_checkMovement_limits_execution_time & " + str(np.round(np.average(self.time_pair_checkMovement_limits_execution_time), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_icp_execution_time & " + str(np.round(np.average(self.time_pair_icp_execution_time), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_checkAndAddKeyframe_execution_time & " + str(np.round(np.average(self.time_pair_checkAndAddKeyframe_execution_time), 5)) + "s " + "\\\\\n"
            timing_log_str += "time_pair_process_frame_execution_time & " + str(np.round(np.average(self.time_pair_process_frame_execution_time), 5)) + "s " + "\\\\\n"


        self.time_pair_run_ids, self.time_pair_run_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_run)
        #self.time_pair_1_ids, self.time_pair_1_execution_times = self.time_keeper.getSyncTimeByFrameID(time_pair_process_new_frame) 

        #calculate time per estimation extrapolation - normal
        extrapolation_ids = self.x_extrapolated_poses_only_2
        normal_ids_mask = np.setdiff1d(self.x, extrapolation_ids)

        #print("Average extrapolation time: ", np.round(np.average(self.time_pair_run_execution_times [extrapolation_ids]), 5), "s")
        #print("Average normal time: ", np.round(np.average(self.time_pair_run_execution_times [normal_ids_mask]), 5), "s")


        timing_log_str += "Whole runtime & " + str(np.round(self.time_keeper.time_save["whole_runtime_done"][0]["time"] - self.time_keeper.time_save["whole_runtime"][0]["time"], 2)) + "s " + "\\\\\n"
        timing_log_str += "Average time per frame & " + str(np.round(np.average(self.time_pair_run_execution_times), 5)) + "s " + "\\\\\n"
        
        if WRITE_LOG:
            with open(timing_log_path, 'w') as datei:
                datei.write(timing_log_str)


    
    def plotMaskResults(self):
        
        # plt.plot(self.x, self.iou_pvnet, label = "IoU PVNet")
        # plt.plot(self.x, self.pixel_acc_pvnet,label  = "Pixel accuracy")
        # plt.plot(self.x, self.precision_acc_pvnet, label = "Precision")
        # plt.plot(self.x, self.recall_pvnet, label = "Recall")
        # plt.plot(self.x, self.dice_pvnet, label = "Dice")
        
        plt.plot(self.x, self.iou_first_mask_pvnet_cutie, label = "IoU Cutie")
        # plt.plot(self.x, self.pixel_acc_first_mask_pvnet_cutie, label = "Pixel accuracy")
        # plt.plot(self.x, self.precision_acc_first_mask_pvnet_cutie, label = "Precision")
        # plt.plot(self.x, self.recall_first_mask_pvnet_cutie, label = "Recall")
        # plt.plot(self.x, self.dice_first_mask_pvnet_cutie, label = "Dice")
        
        #plt.plot(self.x, self.iou_first_mask_offline_cutie, label = "IoU Cutie first mask gt")
        # plt.plot(self.x, self.pixel_acc_first_mask_offline_cutie,label = "Pixel accuracy")
        # plt.plot(self.x, self.precision_acc_first_mask_offline_cutie,  label = "Precision")
        # plt.plot(self.x, self.recall_first_mask_offline_cutie, label = "Recall")
        # plt.plot(self.x, self.dice_first_mask_offline_cutie, label = "Dice")

        #plt.plot(self.x, self.iou_first_mask_pvnet_xmem, label = "IoU XMEM first mask PVNet")
        # plt.plot(self.x, self.pixel_acc_first_mask_pvnet_xmem,label = "Pixel accuracy")
        # plt.plot(self.x, self.precision_acc_first_mask_pvnet_xmem,  label = "Precision")
        # plt.plot(self.x, self.recall_first_mask_pvnet_xmem, label = "Recall")
        # plt.plot(self.x, self.dice_first_mask_pvnet_xmem, label = "Dice")

        #plt.plot(self.x, self.iou_first_mask_offline_xmem, label = "IoU XMEM first mask gt")
        # plt.plot(self.x, self.pixel_acc_first_mask_offline_xmem,label = "Pixel accuracy")
        # plt.plot(self.x, self.precision_acc_first_mask_offline_xmem,  label = "Precision")
        # plt.plot(self.x, self.recall_first_mask_offline_xmem, label = "Recall")
        # plt.plot(self.x, self.dice_first_mask_offline_xmem, label = "Dice")

        ax = plt.gca()
        ax.set_xlabel("Frame-ID")
        ax.set_ylabel("ADD [m] / Value")
        ax.grid(True)
        plt.legend(loc="upper right")
        
        # ax.set_title('IOU comparison', fontsize = 40, fontweight ='bold')
        #plt.show()

    def plotTimingResults(self):
        ax = plt.gca()
        ax.set_xlabel("Frame")
        ax.set_ylabel("Time [s]")

        #plt.plot(self.time_pair_min_match_with_ref_ids, self.time_pair_min_match_with_ref_execution_times, label = "min_match_with_ref")
        #plt.plot(self.time_pair_checkAndAddKeyframe_ids, self.time_pair_checkAndAddKeyframe_execution_times, label = "checkAndAddKeyframe")
        #plt.plot(self.time_pair_pointCloudDenoise_ids, self.time_pair_pointCloudDenoise_execution_times, label = "pointCloudDenoise")
        
        # plt.plot(self.time_pair_invalidatePixelsByMask_ids, self.time_pair_invalidatePixelsByMask_execution_times, label = "invalidatePixelsByMask")
        # plt.plot(self.time_pair_find_corres_ids, self.time_pair_find_corres_execution_times, label = "find_corres")
        # plt.plot(self.time_pair_procrustesByCorrespondence_ids, self.time_pair_procrustesByCorrespondence_execution_times, label = "procrustesByCorrespondence")
        # plt.plot(self.time_pair_selectKeyFramesForBA_ids, self.time_pair_selectKeyFramesForBA_execution_times, label = "selectKeyFramesForBA")
        # plt.plot(self.time_pair_getFeatureMatchPairs_ids, self.time_pair_getFeatureMatchPairs_execution_times, label = "getFeatureMatchPairs")
        # plt.plot(self.time_pair_optimizeGPU_ids, self.time_pair_optimizeGPU_execution_times, label = "optimizeGPU")
        # plt.plot(self.time_pair_process_frame_ids, self.time_pair_process_frame_execution_times, label = "process_frame")
        # plt.plot(self.time_pair_nerf_start_ids, self.time_pair_nerf_start_execution_times, label = "nerf_start")
        # plt.plot(self.time_pair_nerf_pose_adaptation_ids, self.time_pair_nerf_pose_adaptation_execution_times, label = "nerf_pose_adaptation")
        # plt.plot(self.time_pair_rematch_after_nerf_ids, self.time_pair_rematch_after_nerf_execution_times, label = "rematch_after_nerf")

        plt.plot(self.time_pair_invalidatePixelsByMask_ids, self.time_pair_invalidatePixelsByMask_execution_time, label = "invalidatePixelsByMask")
        plt.plot(self.time_pair_denoise_cloud_ids, self.time_pair_denoise_cloud_execution_time, label = "denoise_cloud")
        plt.plot(self.time_pair_find_corres_1_ids, self.time_pair_find_corres_1_execution_time, label = "find_corres_1")
        plt.plot(self.time_pair_selectKeyFramesForBA_ids, self.time_pair_selectKeyFramesForBA_execution_time, label = "selectKeyFramesForBA")
        plt.plot(self.time_pair_getFeatureMatchPairs_ids, self.time_pair_getFeatureMatchPairs_execution_time, label = "getFeatureMatchPairs")
        plt.plot(self.time_pair_find_corres_2_ids, self.time_pair_find_corres_2_execution_time, label = "find_corres_2")
        plt.plot(self.time_pair_optimizeGPU_ids, self.time_pair_optimizeGPU_execution_time, label = "optimizeGPU")
        plt.plot(self.time_pair_checkMovement_limits_ids, self.time_pair_checkMovement_limits_execution_time, label = "checkMovement_limits")
        plt.plot(self.time_pair_checkAndAddKeyframe_ids, self.time_pair_checkAndAddKeyframe_execution_time, label = "checkAndAddKeyframe")
        plt.plot(self.time_pair_process_frame_ids, self.time_pair_process_frame_execution_time, label = "process_frame")
        #plt.plot(self.time_pair_icp_ids, self.time_pair_icp_execution_time, label = "time_pair_icp")



        
        plt.legend(loc="upper right")
        
        #plt.show()

    def setupPlot(self,use_tk_backend = True):
        if use_tk_backend:
            plt.switch_backend('TkAgg')
        plt.rc ('font', size = 15) #20 für masken / 30 für posen / 15 für timing
        fig = plt.figure(figsize=(16, 9), dpi=(1920/16))
        ax = plt.gca()
        ax.set_ylim([0, 1.5]) #1.4 oder 2.5 für Masken / 1.2 oder 1.0 für Posen / 20 oder 1.5für timing 
        ax.set_xlim([0, len(self.x)])

    def plotADDResults(self):
        #x = range(0,len(y))
        #plt.hist(a)
        #matplotlib                3.7.1
        #matplotlib-inline         0.1.7

        
        #plt.style.use(['science','ieee'])
        ax = plt.gca()
        ResultPlotter.x = self.x
        ResultPlotter.y1 = self.add_bundle_orig
        ResultPlotter.y2 = self.add_bundle_feature_matching_spike

        #ResultPlotter.graph1, = ax.plot([0], [0], label="BundleSDF original")
        #ResultPlotter.graph2, = ax.plot([0], [0], label = "Current Implementation")


        #plt.plot(self.x_masked, self.add_pvnet_orig[self.mask], label ="PVNet original")
        #plt.plot(self.x_masked_upnp, self.add_pvnet_upnp[self.mask_upnp],label ="PVNet upnp")
        # plt.plot(self.x_masked_upnp, self.confidence_kpt_0[self.mask_upnp], label ="Confidences kpt 0")
        # plt.plot(self.x_masked_upnp, self.confidence_kpt_1[self.mask_upnp], label ="Confidences kpt 1")
        # plt.plot(self.x_masked_upnp, self.confidence_kpt_2[self.mask_upnp], label ="Confidences kpt 2")
        # plt.plot(self.x_masked_upnp, self.confidence_kpt_3[self.mask_upnp], label ="Confidences kpt 3")
        # plt.plot(self.x_masked_upnp, self.confidence_kpt_4[self.mask_upnp], label ="Confidences kpt 4")
        # plt.plot(self.x_masked_upnp, self.confidence_kpt_5[self.mask_upnp], label ="Confidences kpt 5")
        # plt.plot(self.x_masked_upnp, self.confidence_kpt_6[self.mask_upnp], label ="Confidences kpt 6")
        # plt.plot(self.x_masked_upnp, self.confidence_kpt_7[self.mask_upnp], label ="Confidences kpt 7")
        # plt.plot(self.x_masked_upnp, self.confidence_kpt_8[self.mask_upnp], label ="Confidences kpt 8")
        
        #plt.plot(self.x_masked_upnp, self.avg, label ="Uncertainty avgerage")
        #plt.plot(self.x_masked_upnp, self.stabw, label ="Uncertainty standard deviation")
        
        # plt.plot(self.x, np.ones(self.x.shape) * 0.05)
        # plt.plot(self.x, np.ones(self.x.shape) * 0.65)
        #plt.plot(self.x_masked_upnp, self.stabw, label ="Uncertainty standard deviation")


        #plt.plot(self.x, self.add_bundle_orig, label="Gt segmentation")
        #plt.plot(self.x, self.add_bundle_nonerf, label="No NeRF")
        #plt.plot(self.x, self.add_bundle_nonerf_pvnet, label="First estimation PVNet")
        #plt.plot(x, rot_movement_2, label="Rot movement")
        #plt.plot(self.x_masked, self.rot_movement_2, label="Rot movement")
        #plt.plot(self.x_masked, self.trans_movement_2, label="Trans movement")
        #plt.plot(x, add_bundle_periodic_orig, label="ADD BundleSDF periodic orig")
        #plt.plot(self.x, self.add_bundle_periodic_upnp, label="Periodic PVNet")
        #plt.plot(self.x, self.add_bundle_limit_rot, label="Limit rotation translation")
        # #plt.plot(self.x, self.add_bundle_limit_rot_trans, label="Limit rotation translation")
        #plt.plot(self.x, self.add_bundle_icp, label="ICP")
        # #plt.plot(self.x, self.add_bundle_occ_aware_check_limit, label="ADD BundleSDF Occlusion aware check limits") #1380 problematic -> full occlusion
        # #plt.plot(self.x, self.add_bundle_occ_aware_check_limit_trans_err, label="ADD BundleSDF Occlusion aware trans err") 
        # #plt.plot(self.x, self.add_bundle_occ_aware_check_limit_rot_err, label="ADD BundleSDF Occlusion aware rot err")
        #plt.plot(self.x, self.add_bundle_occ_aware_force_pvnet, label="Occlusion aware") #1380 problematic -> full occlusion
        #plt.plot(self.x,self.add_bundle_feature_matching_spike, label = "Limit feature matching")
        #plt.plot(self.x,self.add_bundle_pose_regression, label = "ADD Pose regression")
        # plt.plot(self.x,self.add_bundle_pose_regression_2, label = "Pose regression 2")
        #plt.plot(self.x,self.add_bundle_pose_regression_minus_4, label = "Pose regression -4")
        # plt.plot(self.x,self.add_bundle_cutie_first_offline_segmentation, label = "Cutie segmentation")
        #plt.plot(self.x,self.add_bundle_orig_cutie_segmentation, label = "Cutie segmentation")
        #plt.plot(self.x,self.add_bundle_orig_xmem_segmentation, label = "XMEM segmentation")
        #plt.plot(self.x,self.add_bundle_pvnet_seg_only, label = "PVNet segmentation")
        #plt.plot(self.x,self.add_bundle_first_pvnet_cutie_segmentation, label = "Cutie first PVNet")
        #plt.plot(self.x,self.add_test, label = "ADD First PVNet Cutie Segmentation_2")
        #plt.plot(self.x,self.add_bundle_current_implementation, label = "Current implementation")
        #plt.plot(self.x,self.add_bundle_union_occlusion, label = "Union occlusion value")

        plt.plot(self.x_extrapolated_poses_only_2, self.add_bundle_extrapolated_poses_only_2, label = "Pose regression 2")
        plt.plot(self.x_extrapolated_poses_only_minus_4, self.add_bundle_extrapolated_poses_only_minus_4, label = "Pose regression -4")




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

    @staticmethod
    def calcADD(x,y, diameter, k_m = 10):
        full_string = ""
        latex_string = ""
        total_add = len(x)
        if total_add > 0:
            for k_m in range(10,50,10):
                k_m = float(k_m) / 100
                good_add = y[y < (k_m * diameter)]
                full_string += f"ADD_{int(k_m * 100)} = " + str (np.round(len(good_add) / total_add,3)) + "\n"
                latex_string += str (np.round(len(good_add) / total_add,3)) + " & "

            k_m = 0.5
            good_add = y[y < (k_m * diameter)]
            full_string += "ADD_50 = " + str (np.round(len(good_add) / total_add,3)) + "\n"
            latex_string += str (np.round(len(good_add) / total_add,3)) + " \\\\\n"
        return full_string, latex_string


if __name__ == "__main__":
    result_plot = ResultPlotter()
    #result_plot.plotADDResults()
    #result_plot.plotMaskResults()
    #result_plot.plotTimingResults()
    #result_plot.exportPlot("plots/BuchVideo/timing/timing_BundleSDF_pose_regression_-4.pdf")
