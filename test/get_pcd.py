import os
import glob

dataset_dir = '../data_collection/asset/dataset'
pcd_root_dir = './pcds'

record_names = os.listdir(pcd_root_dir)
for record_name in record_names:
    shape_id, category, cnt_id, primact_type, trial_id = record_name.split('_')
    partnet_pcd_path = os.path.join(dataset_dir, shape_id, 'point_sample', 'ply-10000.ply')
    pcd_dir = os.path.join(pcd_root_dir, record_name)
    os.system('cp %s %s' % (partnet_pcd_path, pcd_dir)) # copy partnet pcd to pcd_dir
    
    