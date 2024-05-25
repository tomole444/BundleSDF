import open3d as o3d
import cv2

if __name__ == "__main__":
    
    color = "/home/thws_robotik/Documents/Leyh/6dpose/datagen/swot_detection/blenderproc/rc_2023_rgbd_a/out_bop_1/train_pbr/000000/rgb/000000.png"
    depth = "/home/thws_robotik/Documents/Leyh/6dpose/datagen/swot_detection/blenderproc/rc_2023_rgbd_a/out_bop_1/train_pbr/000000/depth/000000.png"

    cameraParam = o3d.io.read_pinhole_camera_parameters("point_clouds/kin_config.json")



    color_raw = o3d.io.read_image(color)
    depth_raw = o3d.io.read_image(depth)
    rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)#, depth_scale=1.0, convert_rgb_to_intensity=False)
    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, o3d.camera.PinholeCameraIntrinsic(
    #    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))#, intrinsic = cameraParam.intrinsic, extrinsic = cameraParam.extrinsic)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd2, intrinsic = cameraParam.intrinsic, extrinsic = cameraParam.extrinsic)
    #pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_raw,intrinsic = cameraParam.intrinsic, extrinsic = cameraParam.extrinsic)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("point_clouds/blenderproc2.ply", pcd)
