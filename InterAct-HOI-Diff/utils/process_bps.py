import torch
import time
from bps_torch.bps import bps_torch
import os
import trimesh
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datasets = ['behave', 'intercap', 'neuraldome', 'omomo', 'grab', 'chairs', 'imhd']
data_root = '/media/volume/InterAct/data'
# initiate the bps module
bps = bps_torch(bps_type='random_uniform',
                n_bps_points=1024,
                radius=1.,
                n_dims=3,
                custom_basis=None)

for dataset in datasets:
    dataset_path = os.path.join(data_root, dataset,'objects')
    save_path = os.path.join(data_root, dataset,'objects_bps')
    os.makedirs(save_path, exist_ok=True)
    # read pointclouds
    object_names = os.listdir(dataset_path)
    for object_name in object_names:
        object_path = os.path.join(dataset_path, object_name, object_name+'.obj')
        pointcloud = trimesh.load(object_path).vertices
        pointcloud = torch.tensor(pointcloud).to(device)
        # encode the pointcloud
        bps_enc = bps.encode(pointcloud,
                                feature_type=['dists','deltas'],
                                x_features=None,
                                custom_basis=None)
        deltas = bps_enc['deltas']
        bps_dec = bps.decode(deltas).squeeze().cpu().numpy()
        # save the encoded features
        object_bps_path = os.path.join(save_path, object_name, object_name+'.npy')
        os.makedirs(os.path.join(save_path, object_name), exist_ok=True)
        np.save(object_bps_path, bps_dec)
        print('Finish encoding object: {}-{}'.format(dataset, object_name))





