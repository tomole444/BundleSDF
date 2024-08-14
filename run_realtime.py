
from bundlesdf import *
import argparse
import os,sys
from tqdm import tqdm
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
from InferenceClient import InferenceClient
import time

def run_video_realtime(video_dir='/home/bowen/debug/2022-11-18-15-10-24_milk', key_folder = '/home/grass/Documents/Leyh/BundleSDF/outBookComb720p', out_folder='/home/bowen/debug/bundlesdf_2022-11-18-15-10-24_milk/', use_segmenter=False, use_gui=False):
  set_seed(0)
  os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}')
  cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml",'r'))
  cfg_bundletrack['SPDLOG'] = int(args.debug_level)
  cfg_bundletrack['depth_processing']["zfar"] = 1.75
  cfg_bundletrack['depth_processing']["percentile"] = 95
  cfg_bundletrack['erode_mask'] = 3
  cfg_bundletrack['debug_dir'] = out_folder+'/'
  cfg_bundletrack['bundle']['max_BA_frames'] = 10
  cfg_bundletrack['bundle']['max_optimized_feature_loss'] = 0.03

  cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 0.3#0.02
  cfg_bundletrack['feature_corres']['max_normal_neighbor'] = 60 #30
  cfg_bundletrack['feature_corres']['max_dist_no_neighbor'] = 0.3 #0.01
  cfg_bundletrack['feature_corres']['max_normal_no_neighbor'] = 60 #20
  # cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 0.02
  # cfg_bundletrack['feature_corres']['max_normal_neighbor'] = 30
  # cfg_bundletrack['feature_corres']['max_dist_no_neighbor'] = 0.01
  # cfg_bundletrack['feature_corres']['max_normal_no_neighbor'] = 20

  cfg_bundletrack['feature_corres']['map_points'] = True
  cfg_bundletrack['feature_corres']['resize'] = 400
  cfg_bundletrack['feature_corres']['rematch_after_nerf'] = True
  cfg_bundletrack['keyframe']['min_rot'] = 5

  cfg_bundletrack['ransac']['inlier_dist'] = 0.2#0.01
  cfg_bundletrack['ransac']['inlier_normal_angle'] = 60#20
  cfg_bundletrack['ransac']['max_trans_neighbor'] = 0.8#0.02
  cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 270#30
  cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 0.8#0.01
  cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 270#10  
  # cfg_bundletrack['ransac']['inlier_dist'] = 0.01
  # cfg_bundletrack['ransac']['inlier_normal_angle'] = 20
  # cfg_bundletrack['ransac']['max_trans_neighbor'] = 0.02
  # cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 30
  # cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 0.01
  # cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 10


  cfg_bundletrack['p2p']['max_dist'] = 0.02
  cfg_bundletrack['p2p']['max_normal_angle'] = 45
  cfg_track_dir = f'{out_folder}/config_bundletrack.yml'
  yaml.dump(cfg_bundletrack, open(cfg_track_dir,'w'))

  cfg_nerf = yaml.load(open(f"{code_dir}/config.yml",'r'))
  cfg_nerf['continual'] = True
  cfg_nerf['trunc_start'] = 0.01
  cfg_nerf['trunc'] = 0.01
  cfg_nerf['mesh_resolution'] = 0.005
  cfg_nerf['down_scale_ratio'] = 1
  cfg_nerf['fs_sdf'] = 0.1
  cfg_nerf['far'] = cfg_bundletrack['depth_processing']["zfar"]
  cfg_nerf['datadir'] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
  cfg_nerf['notes'] = ''
  cfg_nerf['expname'] = 'nerf_with_bundletrack_online'
  cfg_nerf['save_dir'] = cfg_nerf['datadir']
  cfg_nerf_dir = f'{out_folder}/config_nerf.yml'
  yaml.dump(cfg_nerf, open(cfg_nerf_dir,'w'))

  if use_segmenter:
    segmenter = InferenceClient()

  tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5, use_gui=use_gui)
  tracker.loadRelatedKeyFrames(key_folder)
  
  reader = YcbineoatReader(video_dir=video_dir)

  #for i in range(0,len(reader.color_files),args.stride):
  for i in tqdm(range(0,len(reader.color_files),args.stride)):
    color_file = reader.color_files[i]
    color = cv2.imread(color_file)
    H0, W0 = color.shape[:2]
    depth = reader.get_depth(i)
    H,W = depth.shape[:2]
    color = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)

    if i==0:
      mask = reader.get_mask(0)
      mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
      if use_segmenter:
        mask = segmenter.runOffline(color_file.replace('rgb','masks'))
    else:
      if use_segmenter:
        mask = segmenter.runOffline(color_file.replace('rgb','masks'))
      else:
        mask = reader.get_mask(i)
        mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)

    if cfg_bundletrack['erode_mask']>0:
      kernel = np.ones((cfg_bundletrack['erode_mask'], cfg_bundletrack['erode_mask']), np.uint8)
      mask = cv2.erode(mask.astype(np.uint8), kernel)

    id_str = reader.id_strs[i]
    pose_in_model = np.eye(4)

    K = reader.K.copy()
    
    tracker.runRealtime(color, depth, K, id_str, mask=mask, occ_mask=None, pose_in_model=pose_in_model)

  tracker.on_finish()



def draw_pose():
  start = time.time()
  K = np.loadtxt(f'{args.out_folder}/cam_K.txt').reshape(3,3)
  color_files = sorted(glob.glob(f'{args.out_folder}/color/*'))
  #mesh = trimesh.load(f'{args.out_folder}/textured_mesh.obj')
  #to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  #bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  out_dir = f'{args.out_folder}/pose_vis'
  os.makedirs(out_dir, exist_ok=True)
  logging.info(f"Saving to {out_dir}")
  for color_file in color_files:
    color = imageio.imread(color_file)
    pose = np.loadtxt(color_file.replace('.png','.txt').replace('color','ob_in_cam'))
    #logging.info(f"Pose: {pose}")
    #pose = pose@np.linalg.inv(to_origin)
    #vis = draw_posed_3d_box(K, color, ob_in_cam=pose, bbox=bbox, line_color=(255,255,0))
    vis = draw_xyz_axis(color, ob_in_cam=pose, K = K, is_input_rgb = False)
    #vis = draw_posed_3d_box(K, color, ob_in_cam=pose, bbox=bbox, line_color=(255,255,0))
    id_str = os.path.basename(color_file).replace('.png','')
    imageio.imwrite(f'{out_dir}/{id_str}.png', vis)
    logging.info(f"Saving {id_str}.png")
  stop = time.time()
  logging.info(f"Took {stop-start}s") # Takes 192.37s for 681 Pictures



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="run_video", help="run_video/drawpose")
    parser.add_argument('--video_dir', type=str, default="/home/grass/Documents/Leyh/datasets/bookRealtime")
    parser.add_argument('--key_folder', type=str, default="/home/grass/Documents/Leyh/BundleSDF/outBookComb720p")
    parser.add_argument('--out_folder', type=str, default="/home/grass/Documents/Leyh/BundleSDF/outBookRealtime")
    parser.add_argument('--use_segmenter', type=int, default=0)
    parser.add_argument('--use_gui', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1, help='interval of frames to run; 1 means using every frame')
    parser.add_argument('--debug_level', type=int, default=4, help='higher means more logging')
    args = parser.parse_args()

    if args.mode=='run_video':
        run_video_realtime(video_dir=args.video_dir, key_folder=args.key_folder, out_folder=args.out_folder, use_segmenter=args.use_segmenter, use_gui=args.use_gui)
    elif args.mode=='draw_pose':
      draw_pose()
    else:
        raise RuntimeError