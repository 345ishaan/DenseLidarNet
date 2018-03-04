import cPickle as pkl
import numpy as np
import os
import pandas as pd
from ipdb import set_trace as brk
import h5py

bbox_path = '/home/ishan/DenseLidarNet/git/DenseLidarNet/ground_truth/0035_bbox.h5'
idx_path = '/home/ishan/DenseLidarNet/git/DenseLidarNet/ground_truth/0035_idx.h5'
lidar_path = '/home/ishan/DenseLidarNet/git/DenseLidarNet/ground_truth/0035_lidar.h5'


hf_bbox = h5py.File(bbox_path,'r')
hf_idx = h5py.File(idx_path,'r')
hf_lidar = h5py.File(lidar_path,'r')



idx_list = list(hf_idx.get('idx'))
data_path = '/home/ishan/DenseLidarNet/data'

if not os.path.exists(data_path):
	os.makedirs(data_path)

seq_id = 35
pickle_data = []
for idx in idx_list:
	splits = idx[0].split('_')
	fnum = int(splits[2])
	bbox_id = splits[3]
	bbox_data = np.array(hf_bbox.get(idx[1])).tolist()
	lidar_data = np.array(hf_lidar.get(idx[0]))
	x1,y1,x2,y2 = bbox_data
	with open(os.path.join(data_path,idx[0]+'.pickle'),'wb') as f:
		pkl.dump(lidar_data,f,protocol=pkl.HIGHEST_PROTOCOL)
	
	pickle_data.append([seq_id,fnum,x1,y1,x2,y2,os.path.join(data_path,idx[0]+'.pickle')])

with open(os.path.join('all_annt_train.pickle'),'wb') as f:
	pkl.dump(pickle_data,f,protocol=pkl.HIGHEST_PROTOCOL)




