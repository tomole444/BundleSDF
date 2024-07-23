import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    #print(zs.min())
    #print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts


def plot_surface_normals(depth_image, normals):
    plt.figure(figsize=(10, 10))
    #plt.imshow(depth_image, cmap='gray')
    
    color_img = np.zeros((depth_image.shape[0], depth_image.shape[1], 3),dtype = np.uint8)

    plt.title("Surface Normals from Depth Image")
    zy, zx = np.gradient(depth_image)  
    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    #zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)     
    #zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(depth_image)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    #normal *= 255

    plt.imshow(normal)
    plt.show()


plt.switch_backend('TkAgg')

# Load depth image
depth_image_path = '/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/depth/00000.png'  # Replace with your depth image path
K_path = "/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/cam_K.txt"

K = np.loadtxt(K_path)

depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

xyz_mm = rgbd_to_point_cloud(K,depth_image)
xyz_m = xyz_mm / 1e3
depth_scene_pcd = o3d.geometry.PointCloud()
depth_scene_pcd.points = o3d.utility.Vector3dVector(xyz_m)
depth_scene_pcd.estimate_normals()

normals = np.asarray(depth_scene_pcd.normals)
# Compute surface normals


# Plot surface normals
plot_surface_normals(depth_image, normals)


