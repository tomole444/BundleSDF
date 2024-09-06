import pickle
import numpy as np
import cv2

if __name__ == "__main__":
    meta_file = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/HO3D_v3/evaluation/AP10/meta/0000.pkl"
    meta = pickle.load(open(meta_file,'rb'))

    pose = np.load("/home/thws_robotik/Documents/Leyh/6dpose/datasets/HO3D_v3/evaluation/AP10/pose/0.npy")
    glcam_in_cvcam = np.array([[1,0,0,0],
                            [0,-1,0,0],
                            [0,0,-1,0],
                            [0,0,0,1]])
    
    ob_in_cam_gt = np.eye(4)
    ob_in_cam_gt[:3,3] = meta['objTrans']
    ob_in_cam_gt[:3,:3] = cv2.Rodrigues(meta['objRot'].reshape(3))[0]
    ob_in_cam_gt = glcam_in_cvcam@ob_in_cam_gt
    
    print(ob_in_cam_gt)
    print(pose )