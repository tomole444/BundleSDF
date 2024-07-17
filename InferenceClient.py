# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2
import os
import numpy as np
import numpy

import socket
import pickle
import logging

class InferenceClient():
    def __init__(self, cfg_track):

        self.cfg_track = cfg_track

        self.cutie_host= cfg_track["segmenter"]["ip_addr"]
        self.cutie_port= cfg_track["segmenter"]["port"]

        self.pvnet_host= cfg_track["pvnet"]["ip_addr"]
        self.pvnet_port= cfg_track["pvnet"]["port"]

        self.cutie_activated = cfg_track["segmenter"]["activated"]
        self.pvnet_activated = cfg_track["pvnet"]["activated"]

        self.use_pvnet_exclusively = cfg_track["segmenter"]["use_pvnet_exclusively"]

        self.cutie_socket = None
        self.pvnet_socket = None

        if self.cutie_activated:
            logging.info(f"Connecting to Cutie Server {(self.cutie_host, self.cutie_port)}... ")
            self.cutie_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.cutie_socket.connect((self.cutie_host, self.cutie_port))
        if self.pvnet_activated:
            logging.info(f"Connecting to PVNet Server {(self.pvnet_host, self.pvnet_port)}... ")
            self.pvnet_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.pvnet_socket.connect((self.pvnet_host, self.pvnet_port))

    def runOffline(self, mask_file=None):
        return (cv2.imread(mask_file, -1)>0).astype(np.uint8)
    
    def getMask(self, color_img, first_mask_img = None):
        data = None
        mask = None
        if(self.use_pvnet_exclusively):
            data = self.sendPVNetReq(color_img= color_img, request_mask= True)
        else:
            data = self.sendCutieReq(color_img= color_img, first_mask_img= first_mask_img)
        if first_mask_img is None:
            mask = data["mask"]
            if(mask.ndim != 2):
                mask = np.squeeze(mask) 
            mask = np.where(mask >= 1, 1, 0)
        return mask

    def getPVNetPose(self, color_img):
        pvnet_estimation = self.sendPVNetReq(color_img)
        pvnet_ob_in_cam = pvnet_estimation["pose"]
        pvnet_confidences = pvnet_estimation["confidences"].ravel()
        pvnet_confidences = pvnet_confidences[:-1] # dont use last keypoint

        # check if confidence is ok
        pvnet_confidences_avg = np.average(pvnet_confidences)
        pvnet_confidences_std = np.std(pvnet_confidences)
        if pvnet_confidences_std > self.cfg_track["pvnet"]["max_confidence_std"] or pvnet_confidences_avg < self.cfg_track["pvnet"]["min_confidence_avg"] or pvnet_ob_in_cam.round(decimals=6)[2,3] < 0.001:
            logging.info(f"PVNet TF not used pvnet_confidences_std > max_confidence_std: {pvnet_confidences_std > self.cfg_track['pvnet']['max_confidence_std']} \
                         pvnet_confidences_avg < min_confidence_avg: {pvnet_confidences_avg < self.cfg_track['pvnet']['min_confidence_avg']} \
                         pvnet_ob_in_cam.round(decimals=6)[2,3] < 0.001: {pvnet_ob_in_cam.round(decimals=6)[2,3] < 0.001}")
            pvnet_ob_in_cam = None
        return pvnet_ob_in_cam

    def sendPVNetReq(self, color_img, request_mask = False):
        # Pickle the object and send it to the server
        send_data = dict()
        send_data["rgb"] = color_img
        send_data["request_mask"] = request_mask

        
        send_data_pckl = pickle.dumps(send_data)
        
        #sending length first
        self.pvnet_socket.sendall(len(send_data_pckl).to_bytes(4, byteorder='big'))
        self.pvnet_socket.sendall(send_data_pckl)

        ###sending done
        ###receiving start

        #receive datalength
        data_len = int.from_bytes(self.pvnet_socket.recv(4), byteorder='big')
        data = b''
        #receive data
        while len(data) < data_len:
            part = self.pvnet_socket.recv(data_len - len(data))
            data += part

        #data = self.pvnet_socket.recv(4096)
        pvnet_info = pickle.loads(data)
        logging.info(f"TF from PVNet{pvnet_info['pose']}")
        logging.info(f"Confidence from PVNet{pvnet_info['confidences']}")
        logging.info(f"Mask received? {pvnet_info['mask'] is not None}")
        return pvnet_info

    def sendCutieReq(self, color_img, first_mask_img = None):
        # Pickle the object and send it to the server
        send_data = dict()
        if first_mask_img is not None:
            send_data["mask"] = first_mask_img
        else:
            send_data["mask"] = None
        send_data["rgb"] = color_img

        send_data_pckl = pickle.dumps(send_data)
        #sending length first
        self.cutie_socket.sendall(len(send_data_pckl).to_bytes(4, byteorder='big'))
        self.cutie_socket.sendall(send_data_pckl)


        #self.pvnet_socket.sendall(self.pvnet_termination_string)
        #receiving datalength
        data_len = int.from_bytes(self.cutie_socket.recv(4), byteorder='big')
        #rec_data = b''
        data = b''
        while len(data) < data_len:
            part = self.cutie_socket.recv(data_len - len(data))
            data += part
        
        #print("received length",len(data))
        data = pickle.loads(data)
        return data
    
    def closeConnections(self):
        if self.pvnet_activated:
            self.pvnet_socket.close()
        if self.cutie_activated:
            self.cutie_socket.close()