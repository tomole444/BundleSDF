import multiprocessing
import numpy as np

from Utils import draw_xyz_axis
import imageio
import logging
import argparse
import os,sys


try:
  multiprocessing.set_start_method('spawn')
except:
  pass

OUT_FOLDER = "/home/grass/Documents/Leyh/BundleSDF/outFritzCon"

class Visualisations:

    def __init__(self, out_folder):
        self.color_files = sorted(glob.glob(f'{out_folder}/color/*'))
        self.K = np.loadtxt(f'{out_folder}/cam_K.txt').reshape(3,3)
        self.pose_out_dir = f'{out_folder}/pose_vis'
        os.makedirs(self.pose_out_dir, exist_ok=True)
        self.cnt = 0

    def draw_pose(self, color_file):
        color = imageio.imread(color_file)
        pose = np.loadtxt(color_file.replace('.png','.txt').replace('color','ob_in_cam'))
        #logging.info(f"Pose: {pose}")
        #pose = pose@np.linalg.inv(to_origin)
        vis = draw_xyz_axis(color, ob_in_cam=pose, K = self.K, is_input_rgb = False)
        
        return vis
        

    def run_drawpose(self, color_file,lock):
        logging.info(f"Saving to {self.pose_out_dir}")
        #lock.acquire()
        # ...
        # release the lock
        #lock.release()
        #print(f"test {color_file}")
        #
        vis = self.draw_pose(color_file)

        id_str = os.path.basename(color_file).replace('.png','')
        imageio.imwrite(f'{self.pose_out_dir}/{id_str}.png', vis)
        logging.info(f"Saving {id_str}.png")
            

if __name__ == "__main__":

    viz = Visualisations(OUT_FOLDER)

    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock()
    workers = []
    for color_file in viz.color_files:
        worker = multiprocessing.Process(target=viz.run_drawpose, args=(color_file, lock))
        worker.start()
        workers.append(worker)
        
    
    
    for worker in workers:
        worker.join()
    print("Finished")
