# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from nerf_runner import *
from tool import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/BundleTrack/build')
import my_cpp
import operator
from gui import *
from BundleTrack.scripts.data_reader import *
from Utils import *
from loftr_wrapper import LoftrRunner
import multiprocessing,threading
import re
import socket
import pickle
import time
from scipy.spatial.transform import Rotation 
from scipy.spatial import distance
#import cupoch as cph
import open3d as o3d
from velocity_pose_regression import VelocityPoseRegression

try:
  multiprocessing.set_start_method('spawn')
except:
  pass


def run_gui(gui_dict, gui_lock):
  print("GUI started")
  with gui_lock:
    gui = BundleSdfGui(img_height=200)
    gui_dict['started'] = True

  local_dict = {}

  while dpg.is_dearpygui_running():
    with gui_lock:
      if gui_dict['join']:
        break

      for k in ['mesh','color','mask','ob_in_cam','id_str','K','n_keyframe','nerf_num_frames']:
        if k in gui_dict:
          local_dict[k] = gui_dict[k]
          del gui_dict[k]

    if 'nerf_num_frames' in local_dict:
      gui.set_nerf_num_frames(local_dict['nerf_num_frames'])

    if 'mesh' in local_dict:
      logging.info(f"mesh V: {local_dict['mesh'].vertices.shape}")
      gui.update_mesh(local_dict['mesh'])

    if 'color' in local_dict:
      gui.update_frame(rgb=local_dict['color'], mask=local_dict['mask'], ob_in_cam=local_dict['ob_in_cam'], id_str=local_dict['id_str'], K=local_dict['K'], n_keyframe=local_dict['n_keyframe'])

    local_dict = {}

    dpg.render_dearpygui_frame()
    time.sleep(0.03)

  dpg.destroy_context()



def run_nerf(p_dict, kf_to_nerf_list, lock, cfg_nerf, translation, sc_factor, start_nerf_keyframes, use_gui, gui_lock, gui_dict, debug_dir):
  vox_res = 0.01
  nerf_num_frames = 0
  cnt_nerf = -1
  rgbs_all = []
  depths_all = []
  normal_maps_all = []
  masks_all = []
  occ_masks_all = []
  prev_pcd_real_scale = None
  tf_normalize = None
  if translation is not None:
    tf_normalize = np.eye(4)
    tf_normalize[:3,3] = translation
    tf1 = np.eye(4)
    tf1[:3,:3] *= sc_factor
    tf_normalize = tf1@tf_normalize
    cfg_nerf['sc_factor'] = float(sc_factor)
    cfg_nerf['translation'] = translation

  with lock:
    SPDLOG = p_dict['SPDLOG']

  while 1:
    with lock:
      join = p_dict['join']

    if join:
      break

    skip = False
    with lock:
      if cnt_nerf==-1 and len(kf_to_nerf_list)<start_nerf_keyframes:
        skip = True
        p_dict['running'] = False
      else:
        if len(kf_to_nerf_list)>0:
          p_dict['running'] = True
          frame_id = p_dict['frame_id']
          cam_in_obs = p_dict['cam_in_obs'].copy()
          rgbs = []
          depths = []
          normal_maps = []
          masks = []
          occ_masks = []
          for f in kf_to_nerf_list:
            rgbs.append(f['rgb'])
            depths.append(f['depth'])
            masks.append(f['mask'])
            if f['normal_map'] is not None:
              normal_maps.append(f['normal_map'])
            if f['occ_mask'] is not None:
              occ_masks.append(f['occ_mask'])
          K = p_dict['K']
          nerf_num_frames += len(rgbs)
          p_dict['nerf_num_frames'] = nerf_num_frames
          kf_to_nerf_list[:] = []
          if use_gui:
            with gui_lock:
              gui_dict['nerf_num_frames'] = nerf_num_frames
        else:
          skip = True

    if skip:
      time.sleep(0.01)
      continue

    cnt_nerf += 1
    rgbs_all += list(rgbs)
    #print(f"All RGBS {np.array(rgbs_all)}")
    depths_all += list(depths)
    masks_all += list(masks)
    if normal_maps is not None:
      normal_maps_all += list(normal_maps)
    if occ_masks is not None:
      occ_masks_all += list(occ_masks)

    out_dir = f"{debug_dir}/{frame_id}/nerf"
    logging.info(f"out_dir: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    os.system(f"rm -rf {cfg_nerf['datadir']} && mkdir -p {cfg_nerf['datadir']}")

    glcam_in_obs = cam_in_obs@glcam_in_cvcam

    if cfg_nerf['continual']:
      if cnt_nerf==0:
        if translation is None:
          #pdb.set_trace()
          sc_factor,translation,pcd_real_scale, pcd_normalized = compute_scene_bounds(None,glcam_in_obs,K,use_mask=True,base_dir=cfg_nerf['save_dir'],rgbs=np.array(rgbs_all),depths=np.array(depths_all),masks=np.array(masks_all), eps=cfg_nerf['dbscan_eps'], min_samples=cfg_nerf['dbscan_eps_min_samples'])
          sc_factor *= 0.7      # Ensure whole object within bound
          cfg_nerf['sc_factor'] = float(sc_factor)
          cfg_nerf['translation'] = translation
          tf_normalize = np.eye(4)
          tf_normalize[:3,3] = translation
          tf1 = np.eye(4)
          tf1[:3,:3] *= sc_factor
          tf_normalize = tf1@tf_normalize

        pcd_all = pcd_real_scale

      else:
        pcd_all = prev_pcd_real_scale
        for i in range(len(rgbs)):
          pts, colors = compute_scene_bounds_worker(None,K,glcam_in_obs[len(glcam_in_obs)-len(rgbs)+i],use_mask=True,rgb=rgbs[i],depth=depths[i],mask=masks[i])
          pcd_all += toOpen3dCloud(pts, colors)
        pcd_all = pcd_all.voxel_down_sample(vox_res)
        _,keep_mask = find_biggest_cluster(np.asarray(pcd_all.points), eps=cfg_nerf['dbscan_eps'], min_samples=cfg_nerf['dbscan_eps_min_samples'])
        keep_ids = np.arange(len(np.asarray(pcd_all.points)))[keep_mask]
        pcd_all = pcd_all.select_by_index(keep_ids)

        ########## Clear memory
        rgbs_all = []
        depths_all = []
        normal_maps_all = []
        masks_all = []
        occ_masks_all = []

      pcd_normalized = copy.deepcopy(pcd_all)
      pcd_normalized.transform(tf_normalize)
      if normal_maps is not None and len(normal_maps)>0:
        normal_maps = np.array(normal_maps)
      else:
        normal_maps = None
      rgbs,depths,masks,normal_maps,poses = preprocess_data(np.array(rgbs),np.array(depths),np.array(masks),normal_maps=normal_maps,poses=glcam_in_obs,sc_factor=cfg_nerf['sc_factor'],translation=cfg_nerf['translation'])

    else:
      logging.info(f"compute_scene_bounds, latest nerf frame {frame_id}")
      sc_factor,translation,pcd_real_scale, pcd_normalized = compute_scene_bounds(None,glcam_in_obs,K,use_mask=True,base_dir=cfg_nerf['save_dir'],rgbs=np.array(rgbs_all),depths=np.array(depths_all),masks=np.array(masks_all), eps=cfg_nerf['dbscan_eps'], min_samples=cfg_nerf['dbscan_eps_min_samples'])

      cfg_nerf['sc_factor'] = float(sc_factor)
      cfg_nerf['translation'] = translation

      if normal_maps_all is not None and len(normal_maps_all)>0:
        normal_maps = np.array(normal_maps_all)
      else:
        normal_maps = None

      logging.info(f"preprocess_data, latest nerf frame {frame_id}")
      rgbs,depths,masks,normal_maps,poses = preprocess_data(np.array(rgbs_all),np.array(depths_all),np.array(masks_all),normal_maps=normal_maps,poses=glcam_in_obs,sc_factor=cfg_nerf['sc_factor'],translation=cfg_nerf['translation'])

    # cfg_nerf['sampled_frame_ids'] = np.arange(len(rgbs_all))


    if SPDLOG>=2:
      np.savetxt(f"{cfg_nerf['save_dir']}/trainval_poses.txt",glcam_in_obs.reshape(-1,4))
      np.savetxt(f"{debug_dir}/{frame_id}/poses_before_nerf.txt",np.array(cam_in_obs).reshape(-1,4))

    if len(occ_masks_all)>0:
      if cfg_nerf['continual']:
        occ_masks = np.array(occ_masks)
      else:
        occ_masks = np.array(occ_masks_all)
    else:
      occ_masks = None

    if cnt_nerf==0:
      logging.info(f"First nerf run, create Runner, latest nerf frame {frame_id}")
      nerf = NerfRunner(cfg_nerf,rgbs,depths=depths,masks=masks,normal_maps=normal_maps,occ_masks=occ_masks,poses=poses,K=K,build_octree_pcd=pcd_normalized)
    else:
      if cfg_nerf['continual']:
        logging.info(f"add_new_frames, latest nerf frame {frame_id}")
        nerf.add_new_frames(rgbs,depths,masks,normal_maps,poses,occ_masks=occ_masks, new_pcd=pcd_normalized, reuse_weights=False)
      else:
        nerf = NerfRunner(cfg_nerf,rgbs,depths=depths,masks=masks,normal_maps=normal_maps,occ_masks=occ_masks,poses=poses,K=K,build_octree_pcd=pcd_normalized)

    logging.info(f"Start training, latest nerf frame {frame_id}")
    nerf.train()
    logging.info(f"Training done, latest nerf frame {frame_id}")

    optimized_cvcam_in_obs,offset = get_optimized_poses_in_real_world(poses,nerf.models['pose_array'],cfg_nerf['sc_factor'],cfg_nerf['translation'])

    logging.info("Getting mesh")
    mesh = nerf.extract_mesh(isolevel=0,voxel_size=cfg_nerf['mesh_resolution'])
    mesh = mesh_to_real_world(mesh, pose_offset=offset, translation=nerf.cfg['translation'], sc_factor=nerf.cfg['sc_factor'])

    with lock:
      p_dict['optimized_cvcam_in_obs'] = optimized_cvcam_in_obs
      p_dict['running'] = False
      # p_dict['nerf_last'] = nerf    #!NOTE not pickable
      p_dict['mesh'] = mesh

    logging.info(f"nerf done at frame {frame_id}")

    if cfg_nerf['continual']:
      prev_pcd_real_scale = pcd_all.voxel_down_sample(vox_res)

    ####### Log
    if SPDLOG>=2:
      os.system(f"cp -r {cfg_nerf['save_dir']}/image_step_*.png  {out_dir}/")
      with open(f"{out_dir}/config.yml",'w') as ff:
        tmp = copy.deepcopy(cfg_nerf)
        for k in tmp.keys():
          if isinstance(tmp[k],np.ndarray):
            tmp[k] = tmp[k].tolist()
        yaml.dump(tmp,ff)
      shutil.copy(f"{out_dir}/config.yml",f"{cfg_nerf['save_dir']}/")
      np.savetxt(f"{debug_dir}/{frame_id}/poses_after_nerf.txt",np.array(optimized_cvcam_in_obs).reshape(-1,4))
      mesh.export(f"{cfg_nerf['save_dir']}/mesh_real_world.obj")
      os.system(f"rm -rf {cfg_nerf['save_dir']}/step_*_mesh_real_world.obj {cfg_nerf['save_dir']}/*frame*ray*.ply && mv {cfg_nerf['save_dir']}/*  {out_dir}/")




class BundleSdf:
  def __init__(self, cfg_track_dir=f"{code_dir}/config_track.yml", cfg_nerf_dir=f'{code_dir}/config_nerf.yml', start_nerf_keyframes=10, translation=None, sc_factor=None, use_gui=False):
    
    with open(cfg_track_dir,'r') as ff:
      self.cfg_track = yaml.load(ff)
    self.debug_dir = self.cfg_track["debug_dir"]
    self.dataset_dir = self.cfg_track["dataset_dir"]
    self.SPDLOG = self.cfg_track["SPDLOG"]
    self.model_pcd_path = os.path.join(self.dataset_dir,"model_pcd.ply")
    self.start_nerf_keyframes = start_nerf_keyframes
    self.use_gui = use_gui
    self.translation = None
    self.sc_factor = None



    if sc_factor is not None:
      self.translation = translation
      self.sc_factor = sc_factor

    code_dir = os.path.dirname(os.path.realpath(__file__))
    with open(cfg_nerf_dir,'r') as ff:
      self.cfg_nerf = yaml.load(ff)
    

    self.cfg_nerf['notes'] = ''
    self.cfg_nerf['bounding_box'] = np.array(self.cfg_nerf['bounding_box']).reshape(2,3)

    self.manager = multiprocessing.Manager()

    if self.use_gui:
      self.gui_lock = multiprocessing.Lock()
      self.gui_dict = self.manager.dict()
      self.gui_dict['join'] = False
      self.gui_dict['started'] = False
      self.gui_worker = multiprocessing.Process(target=run_gui, args=(self.gui_dict, self.gui_lock))
      self.gui_worker.start()
    else:
      self.gui_lock = None
      self.gui_dict = None

    self.p_dict = self.manager.dict()
    self.kf_to_nerf_list = self.manager.list()
    self.lock = multiprocessing.Lock()
    self.p_dict['running'] = False
    self.p_dict['join'] = False
    self.p_dict['nerf_num_frames'] = 0

    self.p_dict['SPDLOG'] = self.SPDLOG
    
    #activate nerf
    if self.cfg_nerf["activated"]:
      self.p_nerf = multiprocessing.Process(target=run_nerf, args=(self.p_dict, self.kf_to_nerf_list, self.lock, self.cfg_nerf, self.translation, self.sc_factor, start_nerf_keyframes, self.use_gui, self.gui_lock, self.gui_dict, self.debug_dir))
      self.p_nerf.start()



    yml = my_cpp.YamlLoadFile(cfg_track_dir)
    self.bundler = my_cpp.Bundler(yml)
    self.loftr = LoftrRunner()
    self.cnt = -1
    self.K = None
    self.mesh = None

    #Load Model pointcloud for ICP
    self.model_pcd = o3d.io.read_point_cloud(self.model_pcd_path)
    self.model_pcd.estimate_normals()
    
    #frame-data
    self.color = []
    self.depth = []

    #PVNet Server 
    self.pvnet_host = '192.168.99.91'
    self.pvnet_port = 11024
    self.pvnet_socket = None
    self.T_pvnet_bundle = np.identity(4)


    # Data for rotation and translation limitation
    self.last_valid_tf = np.identity(4)
    self.trans_movements = []
    self.rot_movements = []
    self.rot_movement_path = os.path.join(self.debug_dir, "rot_movement")
    self.trans_movement_path = os.path.join(self.debug_dir, "trans_movement")
    self.previous_occluded = 0 # count of previous frames, that have been occluded

    self.continous_discarded_frames = 0 # count of continously discarded frames
    self.last_valid_frames_count = 0 # count of last valid frames

    # Velocity pose estimator
    self.velocity_pose_regression = VelocityPoseRegression(self.K, self.model_pcd_path)
    self.last_tfs = []
    self.last_time_stamp = None
    #self.last_euler_angle = None
    self.last_euler_velocities = []
    self.last_euler_accelerations = []
    self.last_trans_velocities = []
    self.last_trans_accelerations = []

    set_logging_format(log_path= os.path.join(self.debug_dir,"console.log"))



  def on_finish(self):
    if self.use_gui:
      with self.gui_lock:
        self.gui_dict['join'] = True
      self.gui_worker.join()

    with self.lock:
      self.p_dict['join'] = True
    if self.cfg_nerf["activated"]:
      self.p_nerf.join()
      with self.lock:
        if self.p_dict['running']==False and 'optimized_cvcam_in_obs' in self.p_dict:
          for i_f in range(len(self.p_dict['optimized_cvcam_in_obs'])):
            self.bundler._keyframes[i_f]._pose_in_model = self.p_dict['optimized_cvcam_in_obs'][i_f]
            self.bundler._keyframes[i_f]._nerfed = True
          del self.p_dict['optimized_cvcam_in_obs']

  def init_conn_pvnet(self):
    self.pvnet_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.pvnet_socket.connect((self.pvnet_host, self.pvnet_port))

  def close_conn_pvnet(self):
    if self.pvnet_socket is not None:
      self.pvnet_socket.close()

  def make_frame(self, color, depth, K, id_str, mask=None, occ_mask=None, pose_in_model=np.eye(4)):
    H,W = color.shape[:2]
    roi = [0,W-1,0,H-1]
    frame = my_cpp.Frame(color,depth,roi,pose_in_model,self.cnt,id_str,K,self.bundler.yml)
    if mask is not None:
      frame._fg_mask = my_cpp.cvMat(mask)
    if occ_mask is not None:
      frame._occ_mask = my_cpp.cvMat(occ_mask)
    return frame

  def get_pose_from_pvnet(self):
    pvnet_estimation = self.send_image_to_pvnet(self.color)
    pvnet_ob_in_cam = pvnet_estimation["pose"]
    pvnet_confidences = pvnet_estimation["confidences"].ravel()
    pvnet_confidences = pvnet_confidences[:-1] # dont use last keypoint

    # check if confidence is ok
    pvnet_confidences_avg = np.average(pvnet_confidences)
    pvnet_confidences_std = np.std(pvnet_confidences)
    if not(pvnet_confidences_std < self.cfg_track["pvnet"]["max_confidence_std"] and pvnet_confidences_avg > self.cfg_track["pvnet"]["min_confidence_avg"] and pvnet_ob_in_cam.round(decimals=6)[2,3] > 0.001):
      pvnet_ob_in_cam = None
    return pvnet_ob_in_cam


  def find_corres(self, frame_pairs):
    logging.info(f"frame_pairs: {len(frame_pairs)}")
    is_match_ref = len(frame_pairs)==1 and frame_pairs[0][0]._ref_frame_id==frame_pairs[0][1]._id and self.bundler._newframe==frame_pairs[0][0]

    if is_match_ref:
      frameA = frame_pairs[0][0]
      frameB = frame_pairs[0][1]

    imgs, tfs, query_pairs = self.bundler._fm.getProcessedImagePairs(frame_pairs)
    imgs = np.array([np.array(img) for img in imgs])

    if len(query_pairs)==0:
      return
    #pdb.set_trace()
    corres = self.loftr.predict(rgbAs=imgs[::2], rgbBs=imgs[1::2])
    # if not hasattr(self, 'corres'):
    #   self.corres = corres
    # if np.array(corres).shape[1] > np.array(self.corres).shape[1]:
    #   self.corres = corres

    for i_pair in range(len(query_pairs)):
      cur_corres = corres[i_pair][:,:4]
      tfA = np.array(tfs[i_pair*2])
      tfB = np.array(tfs[i_pair*2+1])
      cur_corres[:,:2] = transform_pts(cur_corres[:,:2], np.linalg.inv(tfA))
      cur_corres[:,2:4] = transform_pts(cur_corres[:,2:4], np.linalg.inv(tfB))
      self.bundler._fm._raw_matches[query_pairs[i_pair]] = cur_corres.round().astype(np.uint16)

    min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]
    if is_match_ref and len(self.bundler._fm._raw_matches[frame_pairs[0]])<min_match_with_ref:
      self.bundler._fm._raw_matches[frame_pairs[0]] = []
      self.bundler._newframe._status = my_cpp.Frame.FAIL
      logging.info(f'frame {self.bundler._newframe._id_str} mark FAIL, due to no matching')
      return

    self.bundler._fm.rawMatchesToCorres(query_pairs)
    #pdb.set_trace()
    for pair in query_pairs:
      self.bundler._fm.vizCorresBetween(pair[0], pair[1], 'before_ransac')

    self.bundler._fm.runRansacMultiPairGPU(query_pairs)
    #pdb.set_trace()
    for pair in query_pairs:
      self.bundler._fm.vizCorresBetween(pair[0], pair[1], 'after_ransac')

  def send_image_to_pvnet(self, img):

    # Pickle the object and send it to the server
    img_pckl = pickle.dumps(img)
    self.pvnet_socket.sendall(img_pckl)
    #self.pvnet_socket.sendall(self.pvnet_termination_string)

    data = self.pvnet_socket.recv(4096)
    pvnet_info = pickle.loads(data)
    logging.info(f"TF from PVNet{pvnet_info['pose']}")
    logging.info(f"Confidence from PVNet{pvnet_info['confidences']}")
    return pvnet_info

  def process_new_frame(self, frame):
    logging.info(f"process frame {frame._id_str}")
    
    self.bundler._newframe = frame
    os.makedirs(self.debug_dir, exist_ok=True)
    #ReferenzFrame = letzter Keyframe
    if frame._id>0:
      ref_frame = self.bundler._frames[list(self.bundler._frames.keys())[-1]]
      logging.info(f"Ref Frame: {ref_frame._id}")
      frame._ref_frame_id = ref_frame._id
      frame._pose_in_model = ref_frame._pose_in_model
    else:
      self.bundler._firstframe = frame

    frame.invalidatePixelsByMask(frame._fg_mask)

    #Initiales KOS festlegen
    if frame._id==0 and np.abs(np.array(frame._pose_in_model)-np.eye(4)).max()<=1e-4:
      # Scheitert hieran -> gelöst -> zfar erhöhen
      frame.setNewInitCoordinate()

    n_fg = (np.array(frame._fg_mask)>0).sum()
    if n_fg<100:
      logging.info(f"Frame {frame._id_str} cloud is empty, marked FAIL, roi={n_fg}")
      frame._status = my_cpp.Frame.FAIL;
      self.bundler.forgetFrame(frame)
      return
    
    #Denoising Pointcloud
    if self.cfg_track["depth_processing"]["denoise_cloud"]:
      frame.pointCloudDenoise()
   
    n_valid = frame.countValidPoints()
    n_valid_first = self.bundler._firstframe.countValidPoints()
    if n_valid<n_valid_first/40.0:
      logging.info(f"frame _cloud_down points#: {n_valid} too small compared to first frame points# {n_valid_first}, mark as FAIL")
      frame._status = my_cpp.Frame.FAIL
      self.bundler.forgetFrame(frame)
      return

    
    if frame._id==0:
      self.bundler.checkAndAddKeyframe(frame)   # First frame is always keyframe
      self.bundler._frames[frame._id] = frame
      return
    min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]

    #Suche nach korrespondierenden Frames im Memory 
    self.find_corres([(frame, ref_frame)])
    matches = self.bundler._fm._matches[(frame, ref_frame)]

    if frame._status==my_cpp.Frame.FAIL:
      logging.info(f"find corres fail, mark {frame._id_str} as FAIL")
      self.bundler.forgetFrame(frame)
      return

    matches = self.bundler._fm._matches[(frame, ref_frame)]
    if len(matches)<min_match_with_ref:
      #Falls zu wenige Übereinstimmungen zw. Frame und Ref.-Frame gefunden wurde -> versuche neuen Keyframe mit mehr Übereinstimmungen zu finden
      visibles = []
      for kf in self.bundler._keyframes:
        visible = my_cpp.computeCovisibility(frame, kf)
        visibles.append(visible)
      visibles = np.array(visibles)
      ids = np.argsort(visibles)[::-1]
      found = False
      #pdb.set_trace()
      for id in ids:
        kf = self.bundler._keyframes[id]
        logging.info(f"trying new ref frame {kf._id_str}")
        ref_frame = kf
        frame._ref_frame_id = kf._id
        frame._pose_in_model = kf._pose_in_model
        self.find_corres([(frame, ref_frame)])

        # self.bundler._fm.findCorres(frame, ref_frame)

        if len(self.bundler._fm._matches[(frame,kf)])>=min_match_with_ref:
          logging.info(f"re-choose new ref frame to {kf._id_str}")
          found = True
          break

      if not found:
        frame._status = my_cpp.Frame.FAIL
        logging.info(f"frame {frame._id_str} has not suitable ref_frame, mark as FAIL")
        self.bundler.forgetFrame(frame)
        return


    logging.info(f"frame {frame._id_str} pose update before optimization \n{frame._pose_in_model.round(3)}")
    offset = self.bundler._fm.procrustesByCorrespondence(frame, ref_frame)
    #Pose optimieren aufgrund von Keyframeverschiebung
    frame._pose_in_model = offset@frame._pose_in_model
    logging.info(f"frame {frame._id_str} pose update after optimization \n{frame._pose_in_model.round(3)}")

    #Keyframes vergessen, wenn zu viele
    window_size = self.cfg_track["bundle"]["window_size"]
    if len(self.bundler._frames)-len(self.bundler._keyframes)>window_size:
      for k in self.bundler._frames:
        f = self.bundler._frames[k]
        isforget = self.bundler.forgetFrame(f)
        if isforget:
          logging.info(f"exceed window size, forget frame {f._id_str}")
          break

    self.bundler._frames[frame._id] = frame

    self.bundler.selectKeyFramesForBA()

    local_frames = self.bundler._local_frames

    pairs = self.bundler.getFeatureMatchPairs(self.bundler._local_frames)
    self.find_corres(pairs)
    if frame._status==my_cpp.Frame.FAIL:
      self.bundler.forgetFrame(frame)
      return

    find_matches = False
    self.bundler.optimizeGPU(local_frames, find_matches)

    if frame._status==my_cpp.Frame.FAIL:
      self.bundler.forgetFrame(frame)
      return

    self.bundler.checkAndAddKeyframe(frame)
  
  def process_new_frame_pvnet(self, frame):
    logging.info(f"process frame {frame._id_str}")
    
    self.bundler._newframe = frame
    os.makedirs(self.debug_dir, exist_ok=True)
    #ReferenzFrame = letzter Keyframe
    if frame._id>0:
      #print(f"saved frames {str(list(self.bundler._frames.keys()))}")
      #print(f"saved key-frames {[print(curr_frame._id, ' ',end = '') for curr_frame in self.bundler._keyframes]}")

      ref_frame = self.bundler._frames[list(self.bundler._frames.keys())[-1]]
      logging.info(f"Ref Frame: {ref_frame._id}")
      frame._ref_frame_id = ref_frame._id
      frame._pose_in_model = ref_frame._pose_in_model
      #check if eligible for faster pose estimation
      use_every = self.cfg_track["estimation"]["use_every"]
      use_estimation = False
      if use_every > 0 :
        if frame._id % use_every == 0:
          use_estimation = True
      elif use_every < 0:
        if frame._id % -use_every != 0:
          use_estimation = True
      if use_estimation and self.last_valid_frames_count >= self.cfg_track["estimation"]["use_last"]:
          pose_from_estimator = self.get_Estimation()
          if pose_from_estimator is not None:
            frame._pose_in_model = np.linalg.inv(pose_from_estimator)
            logging.info("using estimator")
            return
    else:
      self.bundler._firstframe = frame
      os.makedirs(self.trans_movement_path, exist_ok=True)
      os.makedirs(self.rot_movement_path, exist_ok=True)

    frame.invalidatePixelsByMask(frame._fg_mask)

    #if(frame._id == 180):
    #  print("here")

    #Initiales KOS festlegen durch PVNet
    if frame._id==0 and np.abs(np.array(frame._pose_in_model)-np.eye(4)).max()<=1e-4:
      # Scheitert hieran -> gelöst -> zfar erhöhen
      #frame.setNewInitCoordinate()
      # Set initial frame with pvnet
      if(self.cfg_track["pvnet"]["activated"]):
        frame.setNewInitCoordinate()
        pvnet_estimation = self.send_image_to_pvnet(self.color)
        pvnet_ob_in_cam = pvnet_estimation["pose"]
        pvnet_confidences = pvnet_estimation["confidences"]
        frame._pose_in_model = np.linalg.inv(pvnet_ob_in_cam)
        # Do icp opitmization
        T_optPose_initialPose = self.optimizeICP(frame)
        frame._pose_in_model = T_optPose_initialPose @ frame._pose_in_model
        #T_cam_pvnet = pvnet_ob_in_cam
        #T_cam_bundle = np.linalg.inv(frame._pose_in_model)
        #self.T_pvnet_bundle = np.linalg.inv(T_cam_pvnet) @ T_cam_bundle
        #frame._pose_in_model = pvnet_pose_in_model
      else:
        frame.setNewInitCoordinate()
    
    n_fg = (np.array(frame._fg_mask)>0).sum()
    if n_fg < self.cfg_track["limits"]["min_mask_pixels"]:
      self.previous_occluded += 1
      logging.info(f"Frame {frame._id_str} cloud is empty, marked FAIL, roi={n_fg}")
      frame._status = my_cpp.Frame.FAIL
      frame._pose_in_model = np.identity(4) #assign invalid pose
      self.bundler.forgetFrame(frame)
      return
   
    
    #Denoising Pointcloud
    if self.cfg_track["depth_processing"]["denoise_cloud"]:
      frame.pointCloudDenoise()
   
    n_valid = frame.countValidPoints()
    n_valid_first = self.bundler._firstframe.countValidPoints()
    if n_valid<n_valid_first/40.0:
      logging.info(f"frame _cloud_down points#: {n_valid} too small compared to first frame points# {n_valid_first}, mark as FAIL")
      frame._status = my_cpp.Frame.FAIL
      self.bundler.forgetFrame(frame)
      return

    if frame._id==0:
      self.bundler.checkAndAddKeyframe(frame)   # First frame is always keyframe
      self.bundler._frames[frame._id] = frame
      return
    elif frame._id % self.cfg_track["pvnet"]["adjust_every"] == 0 or self.previous_occluded > 0:   # check if tf needed from pvnet
      
      pvnet_ob_in_cam = self.get_pose_from_pvnet()

      if pvnet_ob_in_cam is not None and (self.checkMovement(frame,T_cam_obj = pvnet_ob_in_cam) or self.previous_occluded > 0):
        frame._pose_in_model = np.linalg.inv(pvnet_ob_in_cam)
        # Do icp opitmization
        T_optPose_initialPose = self.optimizeICP(frame)
        frame._pose_in_model = T_optPose_initialPose @ frame._pose_in_model
        self.bundler.checkAndAddKeyframe(frame)   # Set frame as keyframe
        self.bundler._frames[frame._id] = frame
        return

    #search for corresponding frame in memory 
    min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]
    self.find_corres([(frame, ref_frame)])
    matches = self.bundler._fm._matches[(frame, ref_frame)]

    if frame._status==my_cpp.Frame.FAIL:
      logging.info(f"find corres fail, mark {frame._id_str} as FAIL")
      self.bundler.forgetFrame(frame)
      frame._pose_in_model = np.identity(4) #assign invalid pose
      return

    matches = self.bundler._fm._matches[(frame, ref_frame)]
    if len(matches)<min_match_with_ref:
      #Falls zu wenige Übereinstimmungen zw. Frame und Ref.-Frame gefunden wurde -> versuche neuen Keyframe mit mehr Übereinstimmungen zu finden
      visibles = []
      for kf in self.bundler._keyframes:
        visible = my_cpp.computeCovisibility(frame, kf)
        visibles.append(visible)
      visibles = np.array(visibles)
      ids = np.argsort(visibles)[::-1]
      found = False
      #pdb.set_trace()
      for id in ids:
        kf = self.bundler._keyframes[id]
        logging.info(f"trying new ref frame {kf._id_str}")
        ref_frame = kf
        frame._ref_frame_id = kf._id
        frame._pose_in_model = kf._pose_in_model
        self.find_corres([(frame, ref_frame)])

        # self.bundler._fm.findCorres(frame, ref_frame)

        if len(self.bundler._fm._matches[(frame,kf)])>=min_match_with_ref:
          logging.info(f"re-choose new ref frame to {kf._id_str}")
          found = True
          break

      if not found:
        frame._status = my_cpp.Frame.FAIL
        logging.info(f"frame {frame._id_str} has not suitable ref_frame, mark as FAIL")
        self.bundler.forgetFrame(frame)
        return


    logging.info(f"frame {frame._id_str} pose update before optimization \n{frame._pose_in_model.round(3)}")
    offset = self.bundler._fm.procrustesByCorrespondence(frame, ref_frame)
    #pose optimization resulting from feature matching -> eliminate spikes 
    feature_matching_optimized_pose = offset@frame._pose_in_model
    distance = np.linalg.norm(feature_matching_optimized_pose[:3,3] - frame._pose_in_model[:3,3])
    
    if distance > self.cfg_track["limits"]["max_feature_matching_offset"]:
      #spike detected -> dont use frame#invalidate frame
      #frame._status = my_cpp.Frame.FAIL
      #frame._pose_in_model = np.identity(4) #assign invalid pose
      pass
    else:
      frame._pose_in_model = feature_matching_optimized_pose
    logging.info(f"frame {frame._id_str} pose update after optimization \n{frame._pose_in_model.round(3)}")

    #Frames vergessen, wenn zu viele
    window_size = self.cfg_track["bundle"]["window_size"]
    if len(self.bundler._frames)-len(self.bundler._keyframes)>window_size:
      for frame_id in self.bundler._frames:
        old_frame = self.bundler._frames[frame_id]
        isforget = self.bundler.forgetFrame(old_frame)
        if isforget:
          logging.info(f"exceed window size, forget frame {old_frame._id_str}")
          break

    self.bundler._frames[frame._id] = frame

    self.bundler.selectKeyFramesForBA()

    local_frames = self.bundler._local_frames

    pairs = self.bundler.getFeatureMatchPairs(self.bundler._local_frames)
    self.find_corres(pairs)

    if n_fg > self.cfg_track["limits"]["min_mask_pixels"]:
      self.previous_occluded = self.previous_occluded - 1 if self.previous_occluded >= 1 else 0

    if frame._status==my_cpp.Frame.FAIL:
      self.bundler.forgetFrame(frame)
      frame._pose_in_model = np.identity(4) #assign invalid pose
      return

    find_matches = False
    self.bundler.optimizeGPU(local_frames, find_matches)


    # limit rot and trans movement
    if not self.checkMovement(frame) and self.previous_occluded == 0 and self.continous_discarded_frames < self.cfg_track["limits"]["force_pvnet_after"]:
      frame._status = my_cpp.Frame.FAIL
    elif self.continous_discarded_frames >= self.cfg_track["limits"]["force_pvnet_after"]:
      pvnet_ob_in_cam = self.get_pose_from_pvnet()
      if pvnet_ob_in_cam is not None:
        frame._pose_in_model = np.linalg.inv(pvnet_ob_in_cam)
      else:
        frame._status = my_cpp.Frame.FAIL

    # Do icp opitmization
    T_optPose_initialPose = self.optimizeICP(frame)
    frame._pose_in_model = T_optPose_initialPose @ frame._pose_in_model

    if frame._status==my_cpp.Frame.FAIL:
      self.bundler.forgetFrame(frame)
      frame._pose_in_model = np.identity(4)
      return
    
    self.bundler.checkAndAddKeyframe(frame)
    # set upper limit to keyframes / sort out keyframes 


  def process_new_frame_realtime(self, frame):
    self.bundler._newframe = frame
    
    frame.invalidatePixelsByMask(frame._fg_mask)

    if frame._id == len(self.bundler._keyframes):
      logging.info(f"first Frame {frame._id}")
      self.bundler._firstframe = frame
      self.bundler._frames[frame._id] = frame
      frame.setNewInitCoordinate()
      return
    

    #Suche nach passendem Keyframe als Refenzframe:
    visibles = []
    for kf in self.bundler._keyframes:
      visible = my_cpp.computeCovisibility(frame, kf)
      visibles.append(visible)
    visibles = np.array(visibles)
    ids = np.argsort(visibles)[::-1]
    min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]
    found = False
    matches_dict = dict()
    #pdb.set_trace()
    for id in ids:
      kf = self.bundler._keyframes[id]
      logging.info(f"trying new ref frame {kf._id_str}")
      ref_frame = kf
      frame._ref_frame_id = kf._id
      frame._pose_in_model = kf._pose_in_model
      #pdb.set_trace()
      self.find_corres([(frame, ref_frame)])
      matches_dict[kf._id] = len(self.bundler._fm._matches[(frame,kf)])

      # self.bundler._fm.findCorres(frame, ref_frame)

      if len(self.bundler._fm._matches[(frame,kf)])>=min_match_with_ref:
        logging.info(f"re-choose new ref frame to {kf._id_str}")
        found = True
        #break
    
    most_matches_id = max(matches_dict.items(), key=operator.itemgetter(1))[0]
    ref_frame = self.bundler._keyframes[most_matches_id]
    frame._ref_frame_id = ref_frame._id
    frame._pose_in_model = ref_frame._pose_in_model
    #pdb.set_trace()
    if not found:
        frame._status = my_cpp.Frame.FAIL
        logging.info(f"frame {frame._id_str} has not suitable ref_frame, mark as FAIL")
        self.bundler.forgetFrame(frame)
        return
    #ReferenzFrame = letzter Keyframe
    # if frame._id>0:
    #   ref_frame = self.bundler._frames[list(self.bundler._frames.keys())[-1]]
    #   logging.info(f"Ref Frame: {ref_frame._id}")
    #   frame._ref_frame_id = ref_frame._id
    #   frame._pose_in_model = ref_frame._pose_in_model
    # else:
    #   self.bundler._firstframe = frame

    frame.invalidatePixelsByMask(frame._fg_mask)

    #Initiales KOS festlegen
    # if frame._id==0 and np.abs(np.array(frame._pose_in_model)-np.eye(4)).max()<=1e-4:
    #   # Scheitert hieran -> gelöst -> zfar erhöhen
    #   frame.setNewInitCoordinate()

    n_fg = (np.array(frame._fg_mask)>0).sum()
    if n_fg<100:
      logging.info(f"Frame {frame._id_str} cloud is empty, marked FAIL, roi={n_fg}")
      frame._status = my_cpp.Frame.FAIL;
      self.bundler.forgetFrame(frame)
      return
    
    #Denoising Pointcloud
    if self.cfg_track["depth_processing"]["denoise_cloud"]:
      frame.pointCloudDenoise()
   
    n_valid = frame.countValidPoints()
    n_valid_first = self.bundler._firstframe.countValidPoints()
    if n_valid<n_valid_first/40.0:
      logging.info(f"frame _cloud_down points#: {n_valid} too small compared to first frame points# {n_valid_first}, mark as FAIL")
      frame._status = my_cpp.Frame.FAIL
      self.bundler.forgetFrame(frame)
      return

    
    if frame._id==0:
      self.bundler.checkAndAddKeyframe(frame)   # First frame is always keyframe
      self.bundler._frames[frame._id] = frame
      return
    min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]

    #Suche nach korrespondierenden Frames im Memory 
    self.find_corres([(frame, ref_frame)])
    matches = self.bundler._fm._matches[(frame, ref_frame)]

    if frame._status==my_cpp.Frame.FAIL:
      logging.info(f"find corres fail, mark {frame._id_str} as FAIL")
      self.bundler.forgetFrame(frame)
      return

    # matches = self.bundler._fm._matches[(frame, ref_frame)]
    # if len(matches)<min_match_with_ref:
    #   #Falls zu wenige Übereinstimmungen zw. Frame und Ref.-Frame gefunden wurde -> versuche neuen Keyframe mit mehr Übereinstimmungen zu finden
    #   visibles = []
    #   for kf in self.bundler._keyframes:
    #     visible = my_cpp.computeCovisibility(frame, kf)
    #     visibles.append(visible)
    #   visibles = np.array(visibles)
    #   ids = np.argsort(visibles)[::-1]
    #   found = False
    #   #pdb.set_trace()
    #   for id in ids:
    #     kf = self.bundler._keyframes[id]
    #     logging.info(f"trying new ref frame {kf._id_str}")
    #     ref_frame = kf
    #     frame._ref_frame_id = kf._id
    #     frame._pose_in_model = kf._pose_in_model
    #     self.find_corres([(frame, ref_frame)])

    #     # self.bundler._fm.findCorres(frame, ref_frame)

    #     if len(self.bundler._fm._matches[(frame,kf)])>=min_match_with_ref:
    #       logging.info(f"re-choose new ref frame to {kf._id_str}")
    #       found = True
    #       break

    #   if not found:
    #     frame._status = my_cpp.Frame.FAIL
    #     logging.info(f"frame {frame._id_str} has not suitable ref_frame, mark as FAIL")
    #     self.bundler.forgetFrame(frame)
    #     return


    logging.info(f"frame {frame._id_str} pose update before optimization \n{frame._pose_in_model.round(3)}")
    offset = self.bundler._fm.procrustesByCorrespondence(frame, ref_frame)
    #Pose optimieren aufgrund von Keyframeverschiebung
    frame._pose_in_model = offset@frame._pose_in_model
    logging.info(f"frame {frame._id_str} pose update after optimization \n{frame._pose_in_model.round(3)}")

    window_size = self.cfg_track["bundle"]["window_size"]
    if len(self.bundler._frames)-len(self.bundler._keyframes)>window_size:
      for k in self.bundler._frames:
        f = self.bundler._frames[k]
        isforget = self.bundler.forgetFrame(f)
        if isforget:
          logging.info(f"exceed window size, forget frame {f._id_str}")
          break

    self.bundler._frames[frame._id] = frame

    self.bundler.selectKeyFramesForBA()

    local_frames = self.bundler._local_frames

    pairs = self.bundler.getFeatureMatchPairs(self.bundler._local_frames)
    self.find_corres(pairs)
    if frame._status==my_cpp.Frame.FAIL:
      self.bundler.forgetFrame(frame)
      return

    find_matches = False
    self.bundler.optimizeGPU(local_frames, find_matches)

    if frame._status==my_cpp.Frame.FAIL:
      self.bundler.forgetFrame(frame)
      return


  def run(self, color, depth, K, id_str, mask=None, occ_mask=None, pose_in_model=np.eye(4)):
    self.cnt += 1

    if self.K is None:
      self.K = K
      with self.lock:
        self.p_dict['K'] = self.K

    if self.use_gui:
      while 1:
        with self.gui_lock:
          started = self.gui_dict['started']
        if not started:
          time.sleep(1)
          logging.info("Waiting for GUI")
          continue
        break

    H,W = color.shape[:2]

    percentile = self.cfg_track['depth_processing']["percentile"]
    print(f"\033[94m depth: {depth.shape} mask: {mask.shape} percentile: {percentile}\033[0m")
    #percentile = 100
    if percentile<100:   # Denoise
      logging.info("percentile denoise start")
      #logging.info(f"depth: {depth.shape}")
      valid = (depth>=0.1) & (mask>0)

      #np.savetxt("test.txt", depth, delimiter = ",")
      #print(f"\033[94m Valid: {valid.shape} \033[0m")

      thres = np.percentile(depth[valid], percentile)
      depth[depth>=thres] = 0
      logging.info("percentile denoise done")
    
    #np.savetxt("test.txt", depth, delimiter = ",")
    #print(f"\033[94m Depth: {depth} \033[0m")

    
    frame = self.make_frame(color, depth, K, id_str, mask, occ_mask, pose_in_model)
    os.makedirs(f"{self.debug_dir}/{frame._id_str}", exist_ok=True)

    logging.info(f"processNewFrame start {frame._id_str}")
    # self.bundler.processNewFrame(frame)
    self.process_new_frame(frame)
    logging.info(f"processNewFrame done {frame._id_str}")

    if self.bundler._keyframes[-1]==frame:
      logging.info(f"{frame._id_str} prepare data for nerf")

      with self.lock:
        self.p_dict['frame_id'] = frame._id_str
        self.p_dict['running'] = True
        self.kf_to_nerf_list.append({
          'rgb': np.array(frame._color).reshape(H,W,3)[...,::-1].copy(),
          'depth': np.array(frame._depth).reshape(H,W).copy(),
          'mask': np.array(frame._fg_mask).reshape(H,W).copy(),
          # 'occ_mask': occ_mask.reshape(H,W),
          # 'normal_map': np.array(frame._normal_map).copy(),
          'occ_mask': None,
          'normal_map': None,
          })
        cam_in_obs = []
        for f in self.bundler._keyframes:
          cam_in_obs.append(np.array(f._pose_in_model).copy())
        self.p_dict['cam_in_obs'] = np.array(cam_in_obs)

      if self.SPDLOG>=2:
        with open(f"{self.debug_dir}/{frame._id_str}/nerf_frames.txt",'w') as ff:
          for f in self.bundler._keyframes:
            ff.write(f"{f._id_str}\n")

      ############# Wait for sync
      while 1:
        with self.lock:
          running = self.p_dict['running']
          nerf_num_frames = self.p_dict['nerf_num_frames']
        if not running:
          break
        if len(self.bundler._keyframes)-nerf_num_frames>=self.cfg_nerf['sync_max_delay']:
          time.sleep(0.01)
          # logging.info(f"wait for sync len(self.bundler._keyframes):{len(self.bundler._keyframes)}, nerf_num_frames:{nerf_num_frames}")
          continue
        break

    rematch_after_nerf = self.cfg_track["feature_corres"]["rematch_after_nerf"]
    logging.info(f"rematch_after_nerf: {rematch_after_nerf}")
    frames_large_update = []
    with self.lock:
      if 'optimized_cvcam_in_obs' in self.p_dict:
        for i_f in range(len(self.p_dict['optimized_cvcam_in_obs'])):
          if rematch_after_nerf:
            trans_update = np.linalg.norm(self.p_dict['optimized_cvcam_in_obs'][i_f][:3,3]-self.bundler._keyframes[i_f]._pose_in_model[:3,3])
            rot_update = geodesic_distance(self.p_dict['optimized_cvcam_in_obs'][i_f][:3,:3], self.bundler._keyframes[i_f]._pose_in_model[:3,:3])
            if trans_update>=0.005 or rot_update>=5/180.0*np.pi:
              frames_large_update.append(self.bundler._keyframes[i_f])
            logging.info(f"{self.bundler._keyframes[i_f]._id_str}, trans_update={trans_update}, rot_update={rot_update}")
          self.bundler._keyframes[i_f]._pose_in_model = self.p_dict['optimized_cvcam_in_obs'][i_f]
          self.bundler._keyframes[i_f]._nerfed = True
        logging.info(f"synced pose from nerf, latest nerf frame {self.bundler._keyframes[len(self.p_dict['optimized_cvcam_in_obs'])-1]._id_str}")
        del self.p_dict['optimized_cvcam_in_obs']

      if self.use_gui:
        with self.gui_lock:
          if 'mesh' in self.p_dict:
            self.gui_dict['mesh'] = self.p_dict['mesh']
            del self.p_dict['mesh']

    if rematch_after_nerf:
      if len(frames_large_update)>0:
        with self.lock:
          nerf_num_frames = self.p_dict['nerf_num_frames']
        logging.info(f"before matches keys: {len(self.bundler._fm._matches)}")
        ks = list(self.bundler._fm._matches.keys())
        for k in ks:
          if k[0] in frames_large_update or k[1] in frames_large_update:
            del self.bundler._fm._matches[k]
            logging.info(f"Delete match between {k[0]._id_str} and {k[1]._id_str}")
        logging.info(f"after matches keys: {len(self.bundler._fm._matches)}")

    self.bundler.saveNewframeResult()
    if self.SPDLOG>=2 and occ_mask is not None:
      os.makedirs(f'{self.debug_dir}/occ_mask/', exist_ok=True)
      cv2.imwrite(f'{self.debug_dir}/occ_mask/{frame._id_str}.png', occ_mask)

    if self.use_gui:
      ob_in_cam = np.linalg.inv(frame._pose_in_model)
      with self.gui_lock:
        self.gui_dict['color'] = color[...,::-1]
        self.gui_dict['mask'] = mask
        self.gui_dict['ob_in_cam'] = ob_in_cam
        self.gui_dict['id_str'] = frame._id_str
        self.gui_dict['K'] = self.K
        self.gui_dict['n_keyframe'] = len(self.bundler._keyframes)

  def runNoNerf(self, color, depth, K, id_str, mask=None, occ_mask=None, pose_in_model=np.eye(4)):
    self.cnt += 1
    self.color = color
    self.depth = depth

    if self.K is None:
      self.K = K
      with self.lock:
        self.p_dict['K'] = self.K

    if self.use_gui:
      while 1:
        with self.gui_lock:
          started = self.gui_dict['started']
        if not started:
          time.sleep(1)
          logging.info("Waiting for GUI")
          continue
        break

    H,W = color.shape[:2]

    percentile = self.cfg_track['depth_processing']["percentile"]
    print(f"\033[94m depth: {depth.shape} mask: {mask.shape} percentile: {percentile}\033[0m")
    #percentile = 100
    if percentile<100:   # Denoise
      logging.info("percentile denoise start")
      #logging.info(f"depth: {depth.shape}")
      valid = (depth>=0.1) & (mask>0)

      #np.savetxt("test.txt", depth, delimiter = ",")
      #print(f"\033[94m Valid: {valid.shape} \033[0m")

      thres = np.percentile(depth[valid], percentile)
      depth[depth>=thres] = 0
      logging.info("percentile denoise done")
    

    
    frame = self.make_frame(color, depth, K, id_str, mask, occ_mask, pose_in_model)
    os.makedirs(f"{self.debug_dir}/{frame._id_str}", exist_ok=True)

    logging.info(f"processNewFrame start {frame._id_str}")
    # self.bundler.processNewFrame(frame)
    #self.process_new_frame(frame)
    self.process_new_frame_pvnet(frame)
    logging.info(f"processNewFrame done {frame._id_str}")

    #correct with pvnet correction
    #logging.info(f"Tranformation difference {self.T_pvnet_bundle}")
    #orig_pose = np.array(frame._pose_in_model).copy()
    #logging.info(f"BundleSDF Pose {orig_pose}")
    #logging.info(f"BundleSDF Inverse-Pose {np.linalg.inv(orig_pose)}")
    #frame._pose_in_model = self.T_pvnet_bundle @ orig_pose #  np.linalg.inv(self.T_pvnet_bundle) 
    
    #logging.info(f"corrected pose Pose {frame._pose_in_model}")
    

    self.bundler.saveNewframeResult()
    #frame._pose_in_model = orig_pose
    
    np.save(os.path.join(self.trans_movement_path, str(frame._id) + ".npy"),  np.array(self.trans_movements))
    np.save(os.path.join(self.rot_movement_path, str(frame._id) + ".npy"),    np.array(self.rot_movements))
    if frame._status != my_cpp.Frame.FAIL:
      self.last_valid_tf = np.linalg.inv(frame._pose_in_model).copy()
      self.continous_discarded_frames = 0
      self.last_valid_frames_count += 1
    else:
      self.continous_discarded_frames += 1
      self.last_valid_frames_count = 0
    if self.SPDLOG>=2 and occ_mask is not None:
      os.makedirs(f'{self.debug_dir}/occ_mask/', exist_ok=True)
      cv2.imwrite(f'{self.debug_dir}/occ_mask/{frame._id_str}.png', occ_mask)

  def runRealtime(self, color, depth, K, id_str, mask=None, occ_mask=None, pose_in_model=np.eye(4)):

    self.cnt += 1

    if self.K is None:
      self.K = K
      with self.lock:
        self.p_dict['K'] = self.K

    if self.use_gui:
      while 1:
        with self.gui_lock:
          started = self.gui_dict['started']
        if not started:
          time.sleep(1)
          logging.info("Waiting for GUI")
          continue
        break

    H,W = color.shape[:2]

    percentile = self.cfg_track['depth_processing']["percentile"]
    print(f"\033[94m depth: {depth.shape} mask: {mask.shape} percentile: {percentile}\033[0m")
    
    if percentile<100:   # Denoise
      logging.info("percentile denoise start")
      valid = (depth>=0.1) & (mask>0)
      thres = np.percentile(depth[valid], percentile)
      depth[depth>=thres] = 0
      logging.info("percentile denoise done")
    


    frame = self.make_frame(color, depth, K, id_str, mask, occ_mask, pose_in_model)
    os.makedirs(f"{self.debug_dir}/{frame._id_str}", exist_ok=True)

    logging.info(f"processNewFrame start {frame._id_str}")
    self.process_new_frame_realtime(frame)
    logging.info(f"processNewFrame done {frame._id_str}")

    ##NERF ANFANG
    # if self.bundler._keyframes[-1]==frame:
    #   logging.info(f"{frame._id_str} prepare data for nerf")

    #   with self.lock:
    #     self.p_dict['frame_id'] = frame._id_str
    #     self.p_dict['running'] = True
    #     self.kf_to_nerf_list.append({
    #       'rgb': np.array(frame._color).reshape(H,W,3)[...,::-1].copy(),
    #       'depth': np.array(frame._depth).reshape(H,W).copy(),
    #       'mask': np.array(frame._fg_mask).reshape(H,W).copy(),
    #       # 'occ_mask': occ_mask.reshape(H,W),
    #       # 'normal_map': np.array(frame._normal_map).copy(),
    #       'occ_mask': None,
    #       'normal_map': None,
    #       })
    #     cam_in_obs = []
    #     for f in self.bundler._keyframes:
    #       cam_in_obs.append(np.array(f._pose_in_model).copy())
    #     self.p_dict['cam_in_obs'] = np.array(cam_in_obs)

    #   if self.SPDLOG>=2:
    #     with open(f"{self.debug_dir}/{frame._id_str}/nerf_frames.txt",'w') as ff:
    #       for f in self.bundler._keyframes:
    #         ff.write(f"{f._id_str}\n")

    #   ############# Wait for sync
    #   while 1:
    #     with self.lock:
    #       running = self.p_dict['running']
    #       nerf_num_frames = self.p_dict['nerf_num_frames']
    #     if not running:
    #       break
    #     if len(self.bundler._keyframes)-nerf_num_frames>=self.cfg_nerf['sync_max_delay']:
    #       time.sleep(0.01)
    #       # logging.info(f"wait for sync len(self.bundler._keyframes):{len(self.bundler._keyframes)}, nerf_num_frames:{nerf_num_frames}")
    #       continue
    #     break

    # rematch_after_nerf = self.cfg_track["feature_corres"]["rematch_after_nerf"]
    # logging.info(f"rematch_after_nerf: {rematch_after_nerf}")
    # frames_large_update = []
    # with self.lock:
    #   if 'optimized_cvcam_in_obs' in self.p_dict:
    #     for i_f in range(len(self.p_dict['optimized_cvcam_in_obs'])):
    #       if rematch_after_nerf:
    #         trans_update = np.linalg.norm(self.p_dict['optimized_cvcam_in_obs'][i_f][:3,3]-self.bundler._keyframes[i_f]._pose_in_model[:3,3])
    #         rot_update = geodesic_distance(self.p_dict['optimized_cvcam_in_obs'][i_f][:3,:3], self.bundler._keyframes[i_f]._pose_in_model[:3,:3])
    #         if trans_update>=0.005 or rot_update>=5/180.0*np.pi:
    #           frames_large_update.append(self.bundler._keyframes[i_f])
    #         logging.info(f"{self.bundler._keyframes[i_f]._id_str}, trans_update={trans_update}, rot_update={rot_update}")
    #       self.bundler._keyframes[i_f]._pose_in_model = self.p_dict['optimized_cvcam_in_obs'][i_f]
    #       self.bundler._keyframes[i_f]._nerfed = True
    #     logging.info(f"synced pose from nerf, latest nerf frame {self.bundler._keyframes[len(self.p_dict['optimized_cvcam_in_obs'])-1]._id_str}")
    #     del self.p_dict['optimized_cvcam_in_obs']

    #   if self.use_gui:
    #     with self.gui_lock:
    #       if 'mesh' in self.p_dict:
    #         self.gui_dict['mesh'] = self.p_dict['mesh']
    #         del self.p_dict['mesh']

    # if rematch_after_nerf:
    #   if len(frames_large_update)>0:
    #     with self.lock:
    #       nerf_num_frames = self.p_dict['nerf_num_frames']
    #     logging.info(f"before matches keys: {len(self.bundler._fm._matches)}")
    #     ks = list(self.bundler._fm._matches.keys())
    #     for k in ks:
    #       if k[0] in frames_large_update or k[1] in frames_large_update:
    #         del self.bundler._fm._matches[k]
    #         logging.info(f"Delete match between {k[0]._id_str} and {k[1]._id_str}")
    #     logging.info(f"after matches keys: {len(self.bundler._fm._matches)}")
    
    ##NERF ENDE

    self.bundler.saveNewframeResult("/home/grass/Documents/Leyh/BundleSDF/outRealtime/")


    if self.use_gui:
      ob_in_cam = np.linalg.inv(frame._pose_in_model)
      with self.gui_lock:
        self.gui_dict['color'] = color[...,::-1]
        self.gui_dict['mask'] = mask
        self.gui_dict['ob_in_cam'] = ob_in_cam
        self.gui_dict['id_str'] = frame._id_str
        self.gui_dict['K'] = self.K
        self.gui_dict['n_keyframe'] = len(self.bundler._keyframes)

  def loadKeyFrames(self, key_folder):
    tmp = sorted(glob.glob(f"{key_folder}/ob_in_cam/*"))
    last_stamp = os.path.basename(tmp[-1]).replace('.txt','')
    logging.info(f'last_stamp {last_stamp}')
    keyframes = yaml.load(open(f'{key_folder}/{last_stamp}/keyframes.yml','r'))
    keys = list(keyframes.keys())
    key_cam_in_obs = []
    for k in keys:
      cam_in_ob = np.array(keyframes[k]['cam_in_ob']).reshape(4,4).astype('float32') 
      key_cam_in_obs.append(cam_in_ob)
    key_cam_in_obs = np.array(key_cam_in_obs)
    logging.info("Starting Loading via cpp")
    K = np.loadtxt(f"{key_folder}/cam_K.txt")
    
    keys = [int(item.replace("keyframe_","")) for item in keys]
    keys = np.array(keys)
    self.bundler.loadKeyframes(keys,5, key_cam_in_obs, K, key_folder,self.bundler.yml)
    
    
    self.cnt = len(keys) - 1
    #pdb.set_trace()
    #logging.info(f"Loaded keyframes#: {len(keyframes)}")

  def run_global_nerf(self, reader=None, get_texture=False, tex_res=1024):
    '''
    @reader: data reader, sometimes we want to use the full resolution raw image
    '''
    self.K = np.loadtxt(f'{self.debug_dir}/cam_K.txt').reshape(3,3)

    tmp = sorted(glob.glob(f"{self.debug_dir}/ob_in_cam/*"))
    last_stamp = os.path.basename(tmp[-1]).replace('.txt','')
    logging.info(f'last_stamp {last_stamp}')
    keyframes = yaml.load(open(f'{self.debug_dir}/{last_stamp}/keyframes.yml','r'))
    logging.info(f"keyframes#: {len(keyframes)}")
    keys = list(keyframes.keys())
    if len(keyframes)>self.cfg_nerf['n_train_image']:
      keys = [keys[0]] + list(np.random.choice(keys, self.cfg_nerf['n_train_image'], replace=False))
      keys = list(set(keys))
      logging.info(f"frame_ids too large, select subset num: {len(keys)}")

    frame_ids = []
    for k in keys:
      frame_ids.append(k.replace('keyframe_',''))

    cam_in_obs = []
    for k in keys:
      cam_in_ob = np.array(keyframes[k]['cam_in_ob']).reshape(4,4)
      cam_in_obs.append(cam_in_ob)
    cam_in_obs = np.array(cam_in_obs)

    out_dir = f"{self.debug_dir}/final/nerf"
    os.system(f"rm -rf {out_dir} && mkdir -p {out_dir}")
    os.system(f'rm -rf {self.debug_dir}/final/used_rgbs/ && mkdir -p {self.debug_dir}/final/used_rgbs/')

    rgbs = []
    rgbNames = []
    depths = []
    normal_maps = []
    masks = []
    occ_masks = []
    for frame_id in frame_ids:
      if reader is not None:
        self.K = reader.K.copy()
        id = reader.id_strs.index(frame_id)
        rgbs.append(reader.get_color(id))
        depths.append(reader.get_depth(id))
        masks.append(reader.get_mask(id))
        #pdb.set_trace()
      else:
        self.cfg_nerf['down_scale_ratio'] = 1   # Images have been downscaled in tracking outputs
        rgb_file = f"{self.debug_dir}/color_segmented/{frame_id}.png"
        shutil.copy(rgb_file, f'{self.debug_dir}/final/used_rgbs/')
        rgb = imageio.imread(rgb_file)
        #pdb.set_trace()
        depth = cv2.imread(rgb_file.replace('color_segmented','depth_filtered'),-1)/1e3
        mask = cv2.imread(rgb_file.replace('color_segmented','mask'),-1)
        rgbs.append(rgb)
        depths.append(depth)
        masks.append(mask)
      rgbNames.append(f"{frame_id}.png")

    glcam_in_obs = cam_in_obs@glcam_in_cvcam

    self.cfg_nerf['sc_factor'] = None
    self.cfg_nerf['translation'] = None

    ######### Reuse normalization
    files = sorted(glob.glob(f"{self.debug_dir}/**/nerf/config.yml", recursive=True))
    if len(files)>0:
      tmp = yaml.load(open(files[-1],'r'))
      self.cfg_nerf['sc_factor'] = float(tmp['sc_factor'])
      self.cfg_nerf['translation'] = np.array(tmp['translation'])

    sc_factor,translation,pcd_real_scale, pcd_normalized = compute_scene_bounds(None,glcam_in_obs,self.K,use_mask=True,base_dir=self.cfg_nerf['save_dir'],rgbs=np.array(rgbs),depths=np.array(depths),masks=np.array(masks), cluster=True, eps=0.01, min_samples=5, sc_factor=self.cfg_nerf['sc_factor'], translation_cvcam=self.cfg_nerf['translation'])

    self.cfg_nerf['sc_factor'] = float(sc_factor)
    self.cfg_nerf['translation'] = translation

    if normal_maps is not None and len(normal_maps)>0:
      normal_maps = np.array(normal_maps)
    else:
      normal_maps = None

    rgbs_raw = np.array(rgbs).copy()
    rgbs,depths,masks,normal_maps,poses = preprocess_data(np.array(rgbs),depths=np.array(depths),masks=np.array(masks),normal_maps=normal_maps,poses=glcam_in_obs,sc_factor=self.cfg_nerf['sc_factor'],translation=self.cfg_nerf['translation'])

    self.cfg_nerf['sampled_frame_ids'] = np.arange(len(rgbs))

    np.savetxt(f"{self.cfg_nerf['save_dir']}/trainval_poses.txt",glcam_in_obs.reshape(-1,4))

    if len(occ_masks)>0:
      occ_masks = np.array(occ_masks)
    else:
      occ_masks = None

    nerf = NerfRunner(self.cfg_nerf,rgbs,depths=depths,masks=masks,normal_maps=normal_maps,occ_masks=occ_masks,poses=poses,K=self.K,build_octree_pcd=pcd_normalized)
    print("Start training")
    nerf.train()
    optimized_cvcam_in_obs,offset = get_optimized_poses_in_real_world(poses,nerf.models['pose_array'],self.cfg_nerf['sc_factor'],self.cfg_nerf['translation'])

    ####### Log
    os.system(f"cp -r {self.cfg_nerf['save_dir']}/image_step_*.png  {out_dir}/")
    with open(f"{out_dir}/config.yml",'w') as ff:
      tmp = copy.deepcopy(self.cfg_nerf)
      for k in tmp.keys():
        if isinstance(tmp[k],np.ndarray):
          tmp[k] = tmp[k].tolist()
      yaml.dump(tmp,ff)
    shutil.copy(f"{out_dir}/config.yml",f"{self.cfg_nerf['save_dir']}/")
    os.system(f"mv {self.cfg_nerf['save_dir']}/*  {out_dir}/ && rm -rf {out_dir}/step_*_mesh_real_world.obj {out_dir}/*frame*ray*.ply")

    torch.cuda.empty_cache()

    np.savetxt(f"{self.debug_dir}/{frame_id}/poses_after_nerf.txt",np.array(optimized_cvcam_in_obs).reshape(-1,4))

    # mesh_files = sorted(glob.glob(f"{self.debug_dir}/final/nerf/step_*_mesh_normalized_space.obj"))
    # mesh = trimesh.load(mesh_files[-1])

    mesh,sigma,query_pts = nerf.extract_mesh(voxel_size=self.cfg_nerf['mesh_resolution'],isolevel=0, return_sigma=True)
    mesh.merge_vertices()
    ms = trimesh_split(mesh, min_edge=100)
    largest_size = 0
    largest = None
    for m in ms:
      # mean = m.vertices.mean(axis=0)
      # if np.linalg.norm(mean)>=0.1*nerf.cfg['sc_factor']:
      #   continue
      if m.vertices.shape[0]>largest_size:
        largest_size = m.vertices.shape[0]
        largest = m
    mesh = largest
    mesh.export(f'{self.debug_dir}/mesh_cleaned.obj')

    if get_texture:
      mesh = nerf.mesh_texture_from_train_images(mesh, rgbs_raw=rgbs_raw, train_texture=False, tex_res=tex_res, rgbNames = rgbNames)

    mesh = mesh_to_real_world(mesh, pose_offset=offset, translation=self.cfg_nerf['translation'], sc_factor=self.cfg_nerf['sc_factor'])
    mesh.export(f'{self.debug_dir}/textured_mesh.obj')

  def optimizeICP(self, frame):
    
    if (not self.cfg_track["icp"]["activated"]):
      return np.identity(4)
    
    intial_pose = np.linalg.inv(frame._pose_in_model)
    mask_img = np.array(frame._fg_mask)
    depth_img = np.array(frame._depth)
    mask_img = mask_img.reshape((mask_img.shape[0],-1))
    depth_img = depth_img.reshape((depth_img.shape[0],-1))

    sem_out = np.where(mask_img>0.8,1,0)
    #sem_out = np.where(radial_out<=max_radii_dm[keypoint_count-1], sem_out,0)

    depth_map = depth_img * sem_out
    xyz_m = rgbd_to_point_cloud(self.K,depth_map)
    #xyz_m = xyz_mm / 1e3
    depth_scene_pcd = o3d.geometry.PointCloud()
    depth_scene_pcd.points = o3d.utility.Vector3dVector(xyz_m)#cph.utility.Vector3fVector(xyz_mm)

    
    #model_pcd.transform(pose)


    depth_scene_pcd.estimate_normals()


    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = self.cfg_track["icp"]["max_iterations"])
    threshold = self.cfg_track["icp"]["threshold"]
    reg_result = None
    if (self.cfg_track["icp"]["type"] == "p2p"):
      reg_result = o3d.pipelines.registration.registration_icp(
            self.model_pcd, depth_scene_pcd, threshold, intial_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria)
    elif (self.cfg_track["icp"]["type"] == "p2l"):
      reg_result = o3d.pipelines.registration.registration_icp(
            self.model_pcd, depth_scene_pcd, threshold, intial_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria)
    print(reg_result)
    optimzed_pose = reg_result.transformation
    T_optPose_initialPose = np.linalg.inv(optimzed_pose) @ intial_pose

    if (self.use_gui):
      self.model_pcd.paint_uniform_color([0, 0.651, 0.929])
      depth_scene_pcd.paint_uniform_color([1, 0.706, 0])

      self.model_pcd.transform(optimzed_pose)
      o3d.visualization.draw_geometries([self.model_pcd, depth_scene_pcd])
      self.model_pcd.transform(np.linalg.inv(optimzed_pose))

    return T_optPose_initialPose

  def checkMovement(self, frame, T_cam_obj = None):
    ret = True
    if T_cam_obj is None:
      T_cam_obj = np.linalg.inv(frame._pose_in_model)
    trans_movement = distance.euclidean(T_cam_obj[:3, 3], self.last_valid_tf[:3,3])
    
    rot_movement = np.sum(np.abs(T_cam_obj[:3, :3] - self.last_valid_tf[:3,:3]))

    # limit rot and trans movement
    if rot_movement > self.cfg_track["limits"]["max_rot_movement"] or trans_movement > self.cfg_track["limits"]["max_t_vec_movement"]:
      ret = False

    

    if(T_cam_obj is None) or (T_cam_obj is not None and ret): 
      self.trans_movements.append(trans_movement)
      self.rot_movements.append(rot_movement)

      if len(self.last_tfs) != 0 and self.last_time_stamp is not None:
        current_time = time.time()
        r = Rotation.from_matrix(T_cam_obj[:3,:3])
        current_angles = r.as_euler("zyx",)
        r = Rotation.from_matrix(self.last_tfs[-1][:3,:3])
        last_angles = r.as_euler("zyx",)
        vel_angle = (current_angles - last_angles) / (current_time - self.last_time_stamp)
        vel_trans = (T_cam_obj[:3,3] - self.last_tfs[-1][:3,3]) / (current_time - self.last_time_stamp)

        if len(self.last_euler_velocities) != 0:
          acc_angle = (vel_angle - self.last_euler_velocities[-1]) / (current_time - self.last_time_stamp)
          acc_trans = (vel_trans - self.last_trans_velocities[-1]) / (current_time - self.last_time_stamp)
          self.last_euler_accelerations.append(acc_angle)
          self.last_trans_accelerations.append(acc_trans)



        self.last_euler_velocities.append(vel_angle)
        self.last_trans_velocities.append(vel_trans)

      self.last_tfs.append(T_cam_obj)
      self.last_time_stamp = time.time()
    return ret

  def get_Estimation(self):
    ret = None
    last_accs = np.array(self.last_euler_accelerations)[len(self.last_euler_accelerations) - self.cfg_track["estimation"]["max_acceleration_std_use_last"] :]
    last_acc_std = np.std(last_accs, axis = 0)  
    if np.all(last_acc_std < self.cfg_track["estimation"]["max_acceleration_std"]):
      last_tfs = np.array(self.last_tfs)[ len(self.last_tfs) - (self.cfg_track["estimation"]["use_last"] - 1) : len(self.last_tfs)]
      last_tfs = np.array(last_tfs)
      ret = self.velocity_pose_regression.predictPose(np.array(last_tfs))
    
    return ret

if __name__=="__main__":
  set_seed(0)
  torch.set_default_tensor_type('torch.cuda.FloatTensor')

  cfg_nerf = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml",'r'))
  cfg_nerf['data_dir'] = '/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/HO3D_v3/evaluation/MPM13'
  cfg_nerf['SPDLOG'] = 1

  cfg_track_dir = '/tmp/config.yml'
  yaml.dump(cfg_nerf, open(cfg_track_dir,'w'))
  tracker = BundleSdf(cfg_track_dir=cfg_track_dir)
  reader = Ho3dReader(tracker.bundler.yml["data_dir"].Scalar())

  os.system(f"rm -rf {tracker.debug_dir} && mkdir -p {tracker.debug_dir}")

  for i,color_file in enumerate(reader.color_files):
    color = cv2.imread(color_file)
    depth = reader.get_depth(i)
    id_str = reader.id_strs[i]
    occ_mask = reader.get_occ_mask(i)
    tracker.run(color, depth, reader.K, id_str, occ_mask=occ_mask)

  print("Done")
