import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.ticker import FuncFormatter
import os
from math import dist
import cv2
import time
import hashlib
import scienceplots
import json
from TimeAnalyser import TimeAnalyser
from scipy.spatial.transform import Rotation 

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
        self.diameter = 0.211
        self.ADD_logpath = "plots/BuchVideo/ADD/ADD-whole.txt"
        self.full_text_add_combined = ""
        self.latex_add_combined = ""

        self.loadADDResults()
        self.loadMaskResults()

        self.time_keeper = None
        self.loadTimingResults()

        self.loadRessourceResults()

        self.loadTensorboardResults()

        self.setupPlot()

        


    def loadADDResults(self):
        
        load_arr = np.load("benchmarks/BuchVideo/ADD_PVNet_orig.npy", allow_pickle=True).item()

        mask = ResultPlotter.calcMask("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/outPVNet239/pose")
        self.add_pvnet_orig = self.loadADDFromFile("benchmarks/BuchVideo/ADD_PVNet_orig.npy", "ADD_PVNet_orig", invalid_poses_mask=~mask)

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



        
        self.mask_upnp = ResultPlotter.calcMask(pose_dir="/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/outPVNet239_upnp/pose")
        self.x_masked_upnp = self.x[self.mask_upnp]
        self.add_pvnet_upnp = self.loadADDFromFile("benchmarks/BuchVideo/ADD_PVNet_upnp.npy", "ADD_PVNet_upnp", invalid_poses_mask= ~ self.mask_upnp)
        
        #self.add_pvnet_upnp[np.invert(self.mask_upnp)] = -1


        self.add_bundle_orig = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_orig.npy", "ADD_BundleSDF_orig")
        
        self.add_bundle_nonerf = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_NoNerf.npy", "ADD_BundleSDF_NoNerf")


        mask = ResultPlotter.calcMask("outBuchVideoFirstMaskPVNet/ob_in_cam")
        self.x_masked_first_pvnet = self.x[mask]
        self.add_bundle_nonerf_pvnet = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_NoNerfPVNet_2.npy", "ADD_BundleSDF_NoNerfPVNet_2")
        self.add_bundle_nonerf_pvnet_gapped = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_NoNerfPVNet_2.npy", "ADD_BundleSDF_NoNerfPVNet_2", invalid_poses_mask= ~mask)
        
        load_arr = np.load("benchmarks/BuchVideo/ADD_Test.npy", allow_pickle=True).item()
        self.add_test = load_arr["result_y"]

        self.add_bundle_periodic_orig = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_PeriodicPVNet_orig.npy", "ADD_BundleSDF_PeriodicPVNet_orig")

        self.add_bundle_periodic_upnp = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_PeriodicPVNet_upnp.npy", "ADD_BundleSDF_PeriodicPVNet_upnp")

        self.add_bundle_limit_rot = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_LimitRot.npy", "ADD_BundleSDF_LimitRot")

        self.add_bundle_limit_rot_trans = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_LimitRotTrans.npy", "ADD_BundleSDF_LimitRotTrans")
        
        mask = ResultPlotter.calcMask("outBuchVideoICP/ob_in_cam")
        self.add_bundle_icp = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_ICP.npy", "ADD_BundleSDF_ICP", invalid_poses_mask=~mask)
        #add_bundle_icp_diff = np.diff(self.add_bundle_icp)
        #self.add_bundle_icp[]

        self.add_bundle_occ_aware = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_Occlusion_Aware.npy", "ADD_BundleSDF_Occlusion_Aware")


        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_Occlusion_Aware_check_limit.npy", allow_pickle=True).item()
        self.add_bundle_occ_aware_check_limit_trans_err = load_arr["trans_err"]
        self.add_bundle_occ_aware_check_limit_rot_err = load_arr["rot_err"]
        self.add_bundle_occ_aware_check_limit = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_Occlusion_Aware_check_limit.npy", "ADD_BundleSDF_Occlusion_Aware_check_limit")

        mask = ResultPlotter.calcMask("outBuchVideoCheckLimit_force_pvnet/ob_in_cam")
        self.add_bundle_occ_aware_force_pvnet = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_Occlusion_Aware_force_pvnet.npy", "ADD_BundleSDF_Occlusion_Aware_force_pvnet", invalid_poses_mask= ~mask)


        mask = ResultPlotter.calcMask("outBuchVideoFeatureOffsetSpike/ob_in_cam")
        self.add_bundle_feature_matching_spike = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_feature_matching_spike.npy", "ADD_BundleSDF_feature_matching_spike", invalid_poses_mask= ~mask)
        

        #mask = ResultPlotter.calcMask("outBuchVideoPoseRegression2/ob_in_cam")
        self.add_bundle_pose_regression_2 = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_pose_regression_2.npy", "ADD_BundleSDF_pose_regression_2")
        self.add_bundle_pose_regression_2 = np.where(self.add_bundle_pose_regression_2 > 0.7, np.nan, self.add_bundle_pose_regression_2)

        self.add_bundle_pose_regression_minus_4 = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_pose_regression_-4.npy", "ADD_BundleSDF_pose_regression_-4")
        self.add_bundle_pose_regression_minus_4 = np.where(self.add_bundle_pose_regression_minus_4 > 0.7, np.nan, self.add_bundle_pose_regression_minus_4)
        
        mask = ResultPlotter.calcMask("outBuchVideoFirstMaskOffline/ob_in_cam")
        self.add_bundle_cutie_first_offline_segmentation = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_cutie_first_offline_segmentation.npy", "ADD_BundleSDF_cutie_first_offline_segmentation", invalid_poses_mask= ~mask)


        self.add_bundle_orig_cutie_segmentation = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_orig_cutie_segmentation.npy", "ADD_BundleSDF_orig_cutie_segmentation")


        self.add_bundle_orig_xmem_segmentation = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_orig_xmem_segmentation.npy", "ADD_BundleSDF_orig_xmem_segmentation")


        self.add_bundle_first_pvnet_cutie_segmentation = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_first_pvnet_cutie_segmentation.npy", "ADD_BundleSDF_first_pvnet_cutie_segmentation")

        mask = ResultPlotter.calcMask("outBuchVideoPVNetSegOnly/ob_in_cam")
        self.add_bundle_pvnet_seg_only = self.loadADDFromFile("benchmarks/BuchVideo/ADD_BundleSDF_pvnet_segmentation_only.npy", "ADD_BundleSDF_pvnet_segmentation_only", invalid_poses_mask= ~mask)


        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_current_implementation.npy", allow_pickle=True).item()
        self.add_bundle_current_implementation = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_union_occlusion.npy", allow_pickle=True).item()
        self.add_bundle_union_occlusion = load_arr["result_y"]

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_extrapolated_poses_only_2.npy", allow_pickle=True).item()
        self.add_bundle_extrapolated_poses_only_2 = load_arr["result_y"]
        self.x_extrapolated_poses_only_2 = load_arr["ids"]
        self.add_bundle_extrapolated_poses_only_2_gapped = np.full_like(self.x, np.nan, dtype=np.float64)
        indices = np.searchsorted(self.x, self.x_extrapolated_poses_only_2)
        self.add_bundle_extrapolated_poses_only_2_gapped[indices] = self.add_bundle_extrapolated_poses_only_2

        load_arr = np.load("benchmarks/BuchVideo/ADD_BundleSDF_extrapolated_poses_only_-4.npy", allow_pickle=True).item()
        self.add_bundle_extrapolated_poses_only_minus_4 = load_arr["result_y"]
        self.x_extrapolated_poses_only_minus_4 = load_arr["ids"]
        self.add_bundle_extrapolated_poses_only_minus_4_gapped = np.full_like(self.x, np.nan, dtype=np.float64)
        indices = np.searchsorted(self.x, self.x_extrapolated_poses_only_minus_4)
        self.add_bundle_extrapolated_poses_only_minus_4_gapped[indices] = self.add_bundle_extrapolated_poses_only_minus_4





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

        
        self.rot_movement_2 = ResultPlotter.calcRotMovement(pose_dir = "outBuchVideoNoNerf/ob_in_cam")
        self.trans_movement_2 = ResultPlotter.calcTransMovement(pose_dir = "outBuchVideoNoNerf/ob_in_cam")


        mask = ResultPlotter.calcMask("/home/thws_robotik/Downloads/outBuchVideoPoseRegression0TimingNoICP/ob_in_cam")
        self.add_bundle_pose_regression_0_no_icp_new = self.loadADDFromFile("benchmarks/BuchVideo/ADD_Bundle_pose_regression_0_no_icp_new.npy", "ADD_Bundle_pose_regression_0_no_icp_new", invalid_poses_mask= ~mask)        
        self.acc_pose_regression_0_ids, self.acc_pose_regression_0_rot, self.acc_pose_regression_0_floating_std_ids, self.acc_pose_regression_0_floating_std = ResultPlotter.calcQuaternionAccs(self.x, "/home/thws_robotik/Downloads/outBuchVideoPoseRegression0TimingNoICP/ob_in_cam")

        self.acc_pose_regression_2_ids, self.acc_pose_regression_2_rot, self.acc_pose_regression_2_floating_std_ids, self.acc_pose_regression_2_floating_std = ResultPlotter.calcQuaternionAccs(self.x, "/home/thws_robotik/Downloads/outBuchVideoPoseRegression2TimingNoICP/ob_in_cam")


        with open(self.ADD_logpath, 'w') as datei:
            datei.write(self.full_text_add_combined)
            datei.write("\n\n\n\n")
            datei.write(self.latex_add_combined)
    
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

    def loadTimingResults(self, timing_file_path = "benchmarks/BuchVideo/time_analysis/timing_pose_regression_-4.npy"):
        self.time_keeper = TimeAnalyser()
        self.time_keeper.load(timing_file_path)
        timing_log_path = "plots/BuchVideo/timing/BundleSDF_pose_regression_-4.txt"
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

        #print("Combined nerf time: ", np.sum(self.time_pair_nerf_start_execution_times))

        timing_log_str += "Whole runtime & " + str(np.round(self.time_keeper.time_save["whole_runtime_done"][0]["time"] - self.time_keeper.time_save["whole_runtime"][0]["time"], 2)) + "s " + "\\\\\n"
        timing_log_str += "Average time per frame & " + str(np.round(np.average(self.time_pair_run_execution_times), 5)) + "s " + "\\\\\n"
        
        if WRITE_LOG:
            with open(timing_log_path, 'w') as datei:
                datei.write(timing_log_str)

    def loadRessourceResults(self):
        load_arr = np.load("benchmarks/BuchVideo/ressource_analysis/ressource_monitor_bundlesdf_orig.npy", allow_pickle=True).item()
        #print(load_arr.keys())
        self.ressources_bundle_sdf_orig_gpu =  []
        self.ressources_bundle_sdf_orig_cpu =  []
        self.ressources_bundle_sdf_orig_memory =  []
        self.ressources_bundle_sdf_orig_ids = []

        for data_point in load_arr["GPU"]:
            self.ressources_bundle_sdf_orig_gpu.append(data_point["usage"] / 1024)
            self.ressources_bundle_sdf_orig_ids.append(data_point["meta"])
        for data_point in load_arr["CPU"]:
            self.ressources_bundle_sdf_orig_cpu.append(data_point["usage"])
        for data_point in load_arr["memory"]:
            self.ressources_bundle_sdf_orig_memory.append(data_point["usage"] / 1024)

        load_arr = np.load("benchmarks/BuchVideo/ressource_analysis/ressource_monitor_pose_regression_0.npy", allow_pickle=True).item()
        self.ressources_pose_regression_0_gpu =  []
        self.ressources_pose_regression_0_cpu =  []
        self.ressources_pose_regression_0_memory =  []
        self.ressources_pose_regression_0_ids = []

        for data_point in load_arr["GPU"]:
            self.ressources_pose_regression_0_gpu.append(data_point["usage"]/ 1024)
            self.ressources_pose_regression_0_ids.append(data_point["meta"])
        for data_point in load_arr["CPU"]:
            self.ressources_pose_regression_0_cpu.append(data_point["usage"])
        for data_point in load_arr["memory"]:
            self.ressources_pose_regression_0_memory.append(data_point["usage"]/ 1024)

        load_arr = np.load("benchmarks/BuchVideo/ressource_analysis/ressource_monitor_pose_regression_2.npy", allow_pickle=True).item()
        #print(load_arr.keys())
        self.ressources_pose_regression_2_gpu =  []
        self.ressources_pose_regression_2_cpu =  []
        self.ressources_pose_regression_2_memory =  []
        self.ressources_pose_regression_2_ids = []

        for data_point in load_arr["GPU"]:
            self.ressources_pose_regression_2_gpu.append(data_point["usage"]/ 1024)
            self.ressources_pose_regression_2_ids.append(data_point["meta"])
        for data_point in load_arr["CPU"]:
            self.ressources_pose_regression_2_cpu.append(data_point["usage"])
        for data_point in load_arr["memory"]:
            self.ressources_pose_regression_2_memory.append(data_point["usage"]/ 1024)

        load_arr = np.load("benchmarks/BuchVideo/ressource_analysis/ressource_monitor_pose_regression_-4.npy", allow_pickle=True).item()
        self.ressources_pose_regression_minus_4_gpu =  []
        self.ressources_pose_regression_minus_4_cpu =  []
        self.ressources_pose_regression_minus_4_memory =  []
        self.ressources_pose_regression_minus_4_ids = []

        for data_point in load_arr["GPU"]:
            self.ressources_pose_regression_minus_4_gpu.append(data_point["usage"]/ 1024)
            self.ressources_pose_regression_minus_4_ids.append(data_point["meta"])
        for data_point in load_arr["CPU"]:
            self.ressources_pose_regression_minus_4_cpu.append(data_point["usage"])
        for data_point in load_arr["memory"]:
            self.ressources_pose_regression_minus_4_memory.append(data_point["usage"]/ 1024)

    def loadTensorboardResults(self):
        dataset_results_path = "benchmarks/ownBuchBig"

        with open(os.path.join(dataset_results_path, 'train_loss.json')) as f:
            d = json.load(f)
        #print(type(d[0][2]))
        self.tensorboard_train_loss_y = np.array(d)[:,2].astype(float)
        self.tensorboard_train_loss_x = np.array(d)[:,1].astype(int)
        
        with open(os.path.join(dataset_results_path, 'train_seg_loss.json')) as f:
            d = json.load(f)
        self.tensorboard_train_seg_loss_y = np.array(d)[:,2].astype(float)
        self.tensorboard_train_seg_loss_x = np.array(d)[:,1].astype(int)

        with open(os.path.join(dataset_results_path, 'train_vote_loss.json')) as f:
            d = json.load(f)
        self.tensorboard_train_vote_loss_y = np.array(d)[:,2].astype(float)
        self.tensorboard_train_vote_loss_x = np.array(d)[:,1].astype(int)

        with open(os.path.join(dataset_results_path, 'val_add.json')) as f:
            d = json.load(f)
        self.tensorboard_val_add_y = np.array(d)[:,2].astype(float)
        self.tensorboard_val_add_x = np.array(d)[:,1].astype(int)

        with open(os.path.join(dataset_results_path, 'val_ap.json')) as f:
            d = json.load(f)
        self.tensorboard_val_ap_y = np.array(d)[:,2].astype(float)
        self.tensorboard_val_ap_x = np.array(d)[:,1].astype(int)

        with open(os.path.join(dataset_results_path, 'val_cmd5.json')) as f:
            d = json.load(f)
        self.tensorboard_val_cmd5_y = np.array(d)[:,2].astype(float)
        self.tensorboard_val_cmd5_x = np.array(d)[:,1].astype(int)

        with open(os.path.join(dataset_results_path, 'val_loss.json')) as f:
            d = json.load(f)
        self.tensorboard_val_loss_y = np.array(d)[:,2].astype(float)
        self.tensorboard_val_loss_x = np.array(d)[:,1].astype(int)

        with open(os.path.join(dataset_results_path, 'val_proj2d.json')) as f:
            d = json.load(f)
        self.tensorboard_val_proj2d_y = np.array(d)[:,2].astype(float)
        self.tensorboard_val_proj2d_x = np.array(d)[:,1].astype(int)

        with open(os.path.join(dataset_results_path, 'val_seg_loss.json')) as f:
            d = json.load(f)
        self.tensorboard_val_seg_loss_y = np.array(d)[:,2].astype(float)
        self.tensorboard_val_seg_loss_x = np.array(d)[:,1].astype(int)

        with open(os.path.join(dataset_results_path, 'val_vote_loss.json')) as f:
            d = json.load(f)
        self.tensorboard_val_vote_loss_y = np.array(d)[:,2].astype(float)
        self.tensorboard_val_vote_loss_x = np.array(d)[:,1].astype(int)

    def setupPlot(self,use_tk_backend = False):
        if use_tk_backend:
            plt.switch_backend('TkAgg')
        plt.rc ('font', size = 30) #20 für masken / 30 für posen / 15 für timing
        fig = plt.figure(figsize=(16, 9), dpi=(1920/16))
        ax = plt.gca()
        ax.set_ylim([0, 1.5]) #1.4 oder 2.5 für Masken / 1.2 oder 1.0 für Posen / 20 oder 1.5 für timing  / 100 für ressource
        ax.set_xlim([0, len(self.x)])
   
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
        
        plt.show()

    def plotRessourceResults(self):
        ax = plt.gca()
        #twin1 = ax.twinx()
        twin2 = ax.twinx()

        #twin2.spines.right.set_position(("axes", 1.07))

        
        # p1, = ax.plot(self.ressources_bundle_sdf_orig_ids, self.ressources_bundle_sdf_orig_cpu, "C0", label="CPU usage")
        # p2, = ax.plot(self.ressources_bundle_sdf_orig_ids, self.ressources_bundle_sdf_orig_gpu, "C1", label="Original GPU-Memory usage")
        # p3, = twin2.plot(self.ressources_bundle_sdf_orig_ids, self.ressources_bundle_sdf_orig_memory, "C2", label="Original Memory usage")

        # p1, = ax.plot(self.ressources_pose_regression_0_ids, self.ressources_pose_regression_0_cpu, "C0", label="CPU usage")
        # p2, = ax.plot(self.ressources_pose_regression_0_ids, self.ressources_pose_regression_0_gpu, "C1", label="Pose regression 0 GPU-Memory usage")
        # p3, = twin2.plot(self.ressources_pose_regression_0_ids, self.ressources_pose_regression_0_memory, "C2", label="Pose regression 0 Memory usage")

        # p1, = ax.plot(self.ressources_pose_regression_2_ids, self.ressources_pose_regression_2_cpu, "C0", label="CPU usage")
        p2, = ax.plot(self.ressources_pose_regression_2_ids, self.ressources_pose_regression_2_gpu, "C1", label="Pose regression 2 GPU-Memory usage")
        p3, = twin2.plot(self.ressources_pose_regression_2_ids, self.ressources_pose_regression_2_memory, "C2", label="Pose regression 2 Memory usage")

        # p1, = ax.plot(self.ressources_pose_regression_minus_4_ids, self.ressources_pose_regression_minus_4_cpu, "C0", label="CPU usage")
        p4, = ax.plot(self.ressources_pose_regression_minus_4_ids, self.ressources_pose_regression_minus_4_gpu, "C3", label="Pose regression -4 GPU-Memory usage")
        p5, = twin2.plot(self.ressources_pose_regression_minus_4_ids, self.ressources_pose_regression_minus_4_memory, "C4", label="Pose regression -4 Memory usage")



        ax.yaxis.label.set_color(p2.get_color())
        # twin1.yaxis.label.set_color(p2.get_color())
        twin2.yaxis.label.set_color(p3.get_color())


        ax.set(xlim=(0, self.x[-1]), ylim=(0, 25), xlabel="Frame-ID", ylabel="GPU-Memory usage [GiB]")
        #twin1.set(ylim=(0, 25), ylabel="GPU-Memory usage [GiB]")
        twin2.set(ylim=(0, 50), ylabel="Memory usage [GiB]")

        ax.tick_params(axis='y', colors=p2.get_color())
        #twin1.tick_params(axis='y', colors=p2.get_color())
        twin2.tick_params(axis='y', colors=p3.get_color())

        ax.legend(handles=[p2, p3, p4, p5])

        ax.grid(True)
        
        plt.show()

    def plotTensorboardResults(self):
        ax = plt.gca()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Proj2D")
        ax.set_xlim([0, 150]) # 500_000
        
        # plt.plot(self.tensorboard_train_loss_x, self.tensorboard_train_loss_y, label = "Train loss")
        #plt.plot(self.tensorboard_train_seg_loss_x, self.tensorboard_train_seg_loss_y, label = "Train segmentation loss")
        #plt.plot(self.tensorboard_train_vote_loss_x, self.tensorboard_train_vote_loss_y, label = "Train vote loss")
        #plt.plot(self.tensorboard_val_add_x, self.tensorboard_val_add_y, label = "Validation ADD")
        #plt.plot(self.tensorboard_val_ap_x, self.tensorboard_val_ap_y, label = "Validation AP")
        #plt.plot(self.tensorboard_val_cmd5_x, self.tensorboard_val_cmd5_y, label = "Validation CMD5")
        #plt.plot(self.tensorboard_val_loss_x, self.tensorboard_val_loss_y, label = "Validation Loss")
        plt.plot(self.tensorboard_val_proj2d_x, self.tensorboard_val_proj2d_y, label = "Validation Proj2D")
        #plt.plot(self.tensorboard_val_seg_loss_x, self.tensorboard_val_seg_loss_y, label = "Validation segmentation loss")
        #plt.plot(self.tensorboard_val_vote_loss_x, self.tensorboard_val_vote_loss_y, label = "Validation vote loss")
        
        plt.legend(loc="lower right")
        ax.grid(True)
        ax.xaxis.set_major_formatter(FuncFormatter(ResultPlotter.format_func_x))
        plt.show()

    def plotADDResults(self):
        #x = range(0,len(y))
        #plt.hist(a)
        #matplotlib                3.7.1
        #matplotlib-inline         0.1.7

        ax = plt.gca()
        ax.set_xlabel("Frame-ID")
        ax.set_ylabel("ADD [m]")# / Trans movement [m] / Rot movement [1]", fontsize = 20)
        #plt.style.use(['science','ieee'])
        ResultPlotter.x = self.x
        ResultPlotter.y1 = self.add_bundle_orig
        ResultPlotter.y2 = self.add_bundle_feature_matching_spike

        #ResultPlotter.graph1, = ax.plot([0], [0], label="BundleSDF original")
        #ResultPlotter.graph2, = ax.plot([0], [0], label = "Current Implementation")

        # PVNet confidence eval
        #plt.plot(self.x, self.add_pvnet_orig, label ="PVNet original")
        #plt.plot(self.x, self.add_pvnet_upnp,label ="PVNet upnp")
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
        
        #plt.plot(self.x_masked_upnp, self.stabw, label ="Uncertainty standard deviation")

        # classic ADD eval
        #plt.plot(self.x, self.add_bundle_orig, label="Gt segmentation")
        #plt.plot(self.x, self.add_bundle_nonerf, label="No NeRF")
        #plt.plot(self.x, self.add_bundle_nonerf_pvnet, label="First estimation PVNet")
        #plt.plot(x, rot_movement_2, label="Rot movement")
        # plt.plot(self.x, self.rot_movement_2, label="Rot movement")
        # plt.plot(self.x, self.trans_movement_2, label="Trans movement")
        #plt.plot(x, add_bundle_periodic_orig, label="ADD BundleSDF periodic orig")
        # plt.plot(self.x, self.add_bundle_periodic_upnp, label="Periodic PVNet")
        #plt.plot(self.x, self.add_bundle_limit_rot, label="Limit rotation translation")
        # #plt.plot(self.x, self.add_bundle_limit_rot_trans, label="Limit rotation translation")
        #plt.plot(self.x, self.add_bundle_icp, label="ICP")
        # #plt.plot(self.x, self.add_bundle_occ_aware_check_limit, label="ADD BundleSDF Occlusion aware check limits") #1380 problematic -> full occlusion
        # #plt.plot(self.x, self.add_bundle_occ_aware_check_limit_trans_err, label="ADD BundleSDF Occlusion aware trans err") 
        # #plt.plot(self.x, self.add_bundle_occ_aware_check_limit_rot_err, label="ADD BundleSDF Occlusion aware rot err")
        #plt.plot(self.x, self.add_bundle_occ_aware_force_pvnet, label="Occlusion aware") #1380 problematic -> full occlusion
        #plt.plot(self.x,self.add_bundle_feature_matching_spike, label = "Limit feature matching")
        #plt.plot(self.x,self.add_bundle_pose_regression, label = "ADD Pose regression")
        #plt.plot(self.x,self.add_bundle_pose_regression_2, label = "Pose regression 2")
        #plt.plot(self.x,self.add_bundle_pose_regression_minus_4, label = "Pose regression -4")
        # plt.plot(self.acc_pose_regression_0_ids, self.acc_pose_regression_0_rot[:,0], label = "q1 x")
        # plt.plot(self.acc_pose_regression_0_ids, self.acc_pose_regression_0_rot[:,1], label = "q2 y")
        # plt.plot(self.acc_pose_regression_0_ids, self.acc_pose_regression_0_rot[:,2], label = "q3 z")
        # plt.plot(self.acc_pose_regression_0_ids, self.acc_pose_regression_0_rot[:,3], label = "q4 w")
        #plt.plot(self.x,self.add_bundle_cutie_first_offline_segmentation, label = "Cutie segmentation")
        #plt.plot(self.x,self.add_bundle_orig_cutie_segmentation, label = "Cutie segmentation")
        #plt.plot(self.x,self.add_bundle_orig_xmem_segmentation, label = "XMEM segmentation")
        #plt.plot(self.x,self.add_bundle_pvnet_seg_only, label = "PVNet segmentation")
        #plt.plot(self.x,self.add_bundle_first_pvnet_cutie_segmentation, label = "Cutie first PVNet")
        #plt.plot(self.x,self.add_test, label = "ADD First PVNet Cutie Segmentation_2")
        #plt.plot(self.x,self.add_bundle_current_implementation, label = "Current implementation")
        #plt.plot(self.x,self.add_bundle_union_occlusion, label = "Union occlusion value")

        #Eval pose regression poses only 
        #plt.plot(self.x_extrapolated_poses_only_2, self.add_bundle_extrapolated_poses_only_2, label = "Pose regression 2")
        #plt.scatter(self.x, self.add_bundle_extrapolated_poses_only_2_gapped, label = "Pose regression 2", s = 10)
        #plt.plot(self.x, self.add_bundle_extrapolated_poses_only_minus_4_gapped, color = "C1", label = "Pose regression -4")

        #Eval / limit rotational accelerations 
        #plt.plot(self.x, self.add_bundle_pose_regression_0_no_icp_new, label = "Pose regression 0")
        # plt.plot(self.acc_pose_regression_0_ids, self.acc_pose_regression_0_rot[:,0], label = "q_{1} x")
        # plt.plot(self.acc_pose_regression_0_ids, self.acc_pose_regression_0_rot[:,1], label = "q_{2} y")
        # plt.plot(self.acc_pose_regression_0_ids, self.acc_pose_regression_0_rot[:,2], label = "q_{3} z")
        # plt.plot(self.acc_pose_regression_0_ids, self.acc_pose_regression_0_rot[:,3], label = "q_{4} w")
        
        plt.plot(self.acc_pose_regression_0_floating_std_ids, np.sum(self.acc_pose_regression_0_floating_std, axis = 1), label = "Standard deviation quaternion")
        plt.plot(self.acc_pose_regression_0_floating_std_ids, np.sum(self.acc_pose_regression_0_floating_std, axis = 1), label = "Standard deviation translation")
        
        ax.set_ylabel("Value")
        

        #display thresholds
        plt.plot(self.x, np.ones(self.x.shape) * 0.05)
        #plt.plot(self.x, np.ones(self.x.shape) * 1)


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

    def loadADDFromFile(self, path:str, name:str, invalid_poses_mask:np.ndarray = None):
        load_arr = np.load(path, allow_pickle=True).item()
        add = load_arr["result_y"]
        full_text_add, latex_add = ResultPlotter.calcADD(load_arr["ids"], load_arr["result_y"], self.diameter)
        self.full_text_add_combined += name + ": " + full_text_add + "\n\n"
        self.latex_add_combined += name + " & " + latex_add
        if invalid_poses_mask is not None:
            add[invalid_poses_mask] = np.nan
        return add

    @staticmethod
    def format_func_x(value, tick_number):
        return f'{int(value):,}'  # Format the tick labels with thousand separator

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
    
    @staticmethod
    def calcQuaternionAccs(x, pose_dir):
        USE_LAST = 20
        
        poses = ResultPlotter.loadPoses(pose_dir)
        poses = np.array(poses)
        rots = poses[:,:3,:3]
        trans = poses[:,:3, 3]
        r = Rotation.from_matrix(rots)
        quaternions = r.as_quat(canonical=True)
        quaternions = np.array(quaternions)
        vels_trans = np.diff(trans, axis = 0)
        vels_rot = np.diff(quaternions, axis = 0)
        accs_trans = np.diff(trans, axis = 0, n = 0)
        accs_rot = np.diff(quaternions, axis = 0, n = 2)
        
        acc_ids = np.arange(len(x))
        acc_ids = acc_ids[2:len(acc_ids)]

        accs_std_ids = []
        accs_rot_std = []
        accs_trans_std = []
        # for idx, pose in enumerate(poses):
        #     if idx == 0:
        #         last_tf = pose
        #         continue
        #     r = Rotation.from_matrix(pose[:3,:3])
        #     current_quat = r.as_quat(canonical=True)
        #     r = Rotation.from_matrix(last_tf[:3,:3])
        #     last_quat = r.as_quat(canonical=True)
        #     vel_quat = (current_quat - last_quat)
        #     vel_trans = (pose[:3,3] - last_tf[:3,3])

        #     if len(vels_rot) != 0:
        #         acc_quat = (vel_quat - vels_rot[-1])
        #         acc_trans = (vel_trans - vels_trans[-1])
        #         accs_rot.append(acc_quat) 
        #         accs_trans.append(acc_trans)
        #         acc_ids.append(x[idx])

        #     vels_rot.append(vel_quat)
        #     vels_trans.append(vel_trans)

        #     last_tf = pose
        accs_rot = np.array(accs_rot)
        acc_ids = np.array(acc_ids)
        for i in range(USE_LAST - 1, len(accs_rot)):
            current_rots = accs_rot[i + 1 - USE_LAST :i + 1]

            current_id = acc_ids[i]
            accs_rot_std.append(np.std(current_rots, axis = 0))
            accs_std_ids.append(current_id)
        
        accs_rot_std = np.array(accs_rot_std)
        accs_std_ids = np.array(accs_std_ids)

        return acc_ids, accs_rot, accs_std_ids, accs_rot_std

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
    #result_plot.plotRessourceResults()
    #result_plot.plotTensorboardResults()
    #result_plot.exportPlot("plots/BuchVideo/vel_est/vel_est_accels_quat_std_0_2_thresh.pdf")
