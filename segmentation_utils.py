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

class Segmenter():
    def __init__(self):
        self.host= "192.168.99.91"#'localhost'
        self.port= 15324
        self.buffer_size = 4096
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def run(self, mask_file=None):
        return (cv2.imread(mask_file, -1)>0).astype(np.uint8)
    
    def runClient(self, color_img, first_mask_img = None):
        # Pickle the object and send it to the server
        send_data = dict()
        if first_mask_img is not None:
            send_data["mask"] = first_mask_img
        else:
            send_data["mask"] = None
        send_data["rgb"] = color_img

        send_data_pckl = pickle.dumps(send_data)
        self.socket.sendall(send_data_pckl)
        #self.pvnet_socket.sendall(self.pvnet_termination_string)
        #receiving datalength
        data_len = int.from_bytes(self.socket.recv(4), byteorder='big')
        #rec_data = b''
        data = b''
        while len(data) < data_len:
            part = self.socket.recv(data_len - len(data))
            data += part
        # while True:
        #     part = self.socket.recv(self.buffer_size)
        #     # if not part: break
        #     rec_data += part
        #     print(len(part))
        #     if len(part) < self.buffer_size:
        #         # End of data (less than buffer_size means no more data left)
        #         break
        
        
        #print("received length",len(data))
        data = pickle.loads(data)
        return data