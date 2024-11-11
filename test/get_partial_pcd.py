import os
import numpy as np
import random
import torch
import cv2
from PIL import Image
import open3d as o3d
import argparse
import json

from env_ori import Env
from camera import Camera

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')

parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--record_name', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--use_mask', type=str, help='whether use movable mask')
eval_conf = parser.parse_args()

## other args
manipllm_out_dir = './test_results/result_ori'
pcd_out_dir = eval_conf.out_dir #'./partial_pcds'
os.makedirs(pcd_out_dir, exist_ok=True)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


## previous info are saved in result.json
shape_id, category, cnt_id, primact_type, trial_id = eval_conf.record_name.split('_')
out_dir = os.path.join(pcd_out_dir, '%s_%s_%s_%s_%d' % (shape_id, category, cnt_id, primact_type, int(trial_id)))
os.makedirs(out_dir, exist_ok=True)
# out_name = '%s_%s_%s_%s_%d' % (shape_id, category, cnt_id, primact_type, int(trial_id))

flog = open(os.path.join(manipllm_out_dir, 'log.txt'), 'w')
out_info = dict()
try:
    with open(os.path.join(eval_conf.data_dir, eval_conf.record_name, 'result.json'), 'r') as fin:
        replay_data = json.load(fin)
except:
    print('no replay data')
    exit(1)


## consturct environment
env = Env(flog=flog, show_gui=(not eval_conf.no_gui))

# setup camera
cam_theta = replay_data['camera_metadata']['theta']
cam_phi = replay_data['camera_metadata']['phi']
cam_dist = replay_data['camera_metadata']['dist']
cam = Camera(env, theta=cam_theta, phi=cam_phi, dist=cam_dist)
out_info['camera_metadata_init'] = cam.get_metadata_json()

if not eval_conf.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam_theta, -cam_phi)

## load shape
object_urdf_fn = '../data_collection/asset/dataset/%s/mobility.urdf' % shape_id
flog.write('object_urdf_fn: %s\n' % object_urdf_fn)
object_material = env.get_material(4, 4, 0.01)
state = replay_data['object_state']
flog.write('Object State: %s\n' % state)
out_info['object_state'] = state
scale = replay_data['scale']
env.load_object(scale, object_urdf_fn, object_material, state=state)

joint_angles = replay_data['joint_angles']
env.set_object_joint_angles(joint_angles)
out_info['joint_angles'] = joint_angles
out_info['joint_angles_lower'] = env.joint_angles_lower
out_info['joint_angles_upper'] = env.joint_angles_upper
cur_qpos = env.get_object_qpos()


# simulate some steps for the object to stay rest
still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 5000 and wait_timesteps < 20000:
    env.step()
    env.render()
    cur_new_qpos = env.get_object_qpos()
    invalid_contact = False
    for c in env.scene.get_contacts():
        for p in c.points:
            if abs(p.impulse @ p.impulse) > 1e-4:
                invalid_contact = True
                break
        if invalid_contact:
            break
    if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
        still_timesteps += 1
    else:
        still_timesteps = 0
    cur_qpos = cur_new_qpos
    wait_timesteps += 1

if still_timesteps < 5000:
    printout(flog, 'Object Not Still!')
    flog.close()
    env.close()
    exit(1)

## generate point cloud

# get color and depth observations
# rgb: [H, W, 3], depth: [H, W]
rgb, depth = cam.get_observation()
rgb_viz = (rgb * 255).astype(np.uint8)
Image.fromarray(rgb_viz).save(os.path.join(out_dir, 'rgb_img.png'))
img = Image.fromarray(rgb_viz)

# get moveable mask
object_link_ids = env.movable_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
mask = (gt_movable_link_mask > 0)
Image.fromarray((mask*255).astype(np.uint8)).save(os.path.join(out_dir, 'mask_img.png'))

# generate point cloud in world coordinate
# cam_XYZA_pts: [N, 4], cam_XYZA_id1: [N], cam_XYZA_id2: [N], out: [996, 996, 4]
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, out = cam.compute_camera_XYZA(depth)

# save point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cam_XYZA_pts)
pcd.colors = o3d.utility.Vector3dVector(rgb[cam_XYZA_id1, cam_XYZA_id2])
o3d.io.write_point_cloud(os.path.join(out_dir, 'pcd.ply'), pcd)

# save the masked point cloud
pcd_masked = o3d.geometry.PointCloud()
pcd_masked.points = o3d.utility.Vector3dVector(cam_XYZA_pts[mask[cam_XYZA_id1, cam_XYZA_id2]])
pcd_masked.colors = o3d.utility.Vector3dVector(rgb[cam_XYZA_id1, cam_XYZA_id2][mask[cam_XYZA_id1, cam_XYZA_id2]])
o3d.io.write_point_cloud(os.path.join(out_dir, 'pcd_masked.ply'), pcd_masked)

