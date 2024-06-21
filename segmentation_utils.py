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

from os import path
import logging
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
from hydra import compose, initialize

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from Cutie.cutie.inference.data.vos_test_dataset import VOSTestDataset
from Cutie.cutie.inference.data.burst_test_dataset import BURSTTestDataset
from Cutie.cutie.model.cutie import CUTIE
from Cutie.cutie.inference.inference_core import InferenceCore
from Cutie.cutie.inference.utils.results_utils import ResultSaver, make_zip
from Cutie.cutie.inference.utils.burst_utils import BURSTResultHandler
from Cutie.cutie.inference.utils.args_utils import get_dataset_cfg
from Cutie.gui.interactive_utils import image_to_torch, torch_prob_to_numpy_mask, index_numpy_to_one_hot_torch, overlay_davis

class Segmenter():
    def __int__(self):
        with torch.inference_mode():
            initialize(version_base='1.3.2', config_path="cutie/config", job_name="eval_config")
            cfg = compose(config_name="eval_config")

            with open_dict(cfg):
                cfg['weights'] = './weights/cutie-base-mega.pth'

            data_cfg = get_dataset_cfg(self.cfg)

            # Load the network weights
            self.cutie = CUTIE(self.cfg).cuda().eval()
            model_weights = torch.load(self.cfg.weights)
            self.cutie.load_weights(self.model_weights)
            self.processor = InferenceCore(self.cutie, cfg=self.cfg)

        return

    def run(self, mask_file=None):
        return (cv2.imread(mask_file, -1)>0).astype(np.uint8)
    
    def setFirstMask(self, first_mask, first_color_img):
        device = 'cuda'
        torch.cuda.empty_cache()
        self.num_objects = len(np.unique(first_mask)) - 1
        mask_torch = index_numpy_to_one_hot_torch(first_mask, self.num_objects+1).to(device)
        # convert numpy array to pytorch tensor format
        frame_torch = image_to_torch(first_color_img, device=device)
        with torch.inference_mode():
            # the background mask is not fed into the model
            prediction = self.processor.step(frame_torch, mask_torch[1:], idx_mask=False)
        

    def runCutie(self, color_img):
        
        device = 'cuda'
        torch.cuda.empty_cache()
        mask_img = None
        with torch.inference_mode():

            # convert numpy array to pytorch tensor format
            frame_torch = image_to_torch(color_img, device=device)
            # propagate only
            prediction = self.processor.step(frame_torch)

            # argmax, convert to numpy
            mask_img = torch_prob_to_numpy_mask(prediction)
        return mask_img
