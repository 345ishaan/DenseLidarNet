import sys
if sys.version_info >= (3,0):
	import _pickle as pkl
else:
	import cPickle as pkl
import numpy as np
import os
import pandas as pd
from pdb import set_trace as brk
import h5py


def process_gt_data(seq_id, data_path):

	bbox_path = '../ground_truth/' + seq_id + '_bbox.h5'
	idx_path = '../ground_truth/' + seq_id + '_idx.h5'
	lidar_path = '../ground_truth/' + seq_id + '_lidar.h5'
	
	print (lidar_path)
	hf_bbox = h5py.File(bbox_path,'r')
	hf_idx = h5py.File(idx_path,'r')
	hf_lidar = h5py.File(lidar_path,'r')

	idx_list = list(hf_idx.get('idx'))

	if not os.path.exists(data_path):
		os.makedirs(data_path)


	pickle_data = []
	for idx in idx_list:
		splits = idx[0].decode().split('_')
		fnum = int(splits[2])
		bbox_id = splits[3]

		bbox_data = np.array(hf_bbox.get(idx[1])).tolist()
		lidar_data = np.array(hf_lidar.get(idx[0]+'/lidar_pts'))
		tf_data = np.array(hf_lidar.get(idx[0]+'/tf_pts'))
		center_data = np.array(hf_lidar.get(idx[0]+'/center'))
		dims_data = np.array(hf_lidar.get(idx[0]+'/dims'))

		x1,y1,x2,y2 = bbox_data
		with open(os.path.join(data_path,idx[0]+'_lidar_pts.pickle'),'wb') as f:
			pkl.dump(lidar_data,f,protocol=pkl.HIGHEST_PROTOCOL)
	
		with open(os.path.join(data_path,idx[0]+'_tf_pts.pickle'),'wb') as f:
			pkl.dump(tf_data,f,protocol=pkl.HIGHEST_PROTOCOL)

		pickle_data.append([seq_id,fnum,x1,y1,x2,y2,os.path.join(data_path,idx[0]+'_lidar_pts.pickle'),os.path.join(data_path,idx[0]+'_tf_pts.pickle')])
	return pickle_data

if __name__ == '__main__':

	seq_ids = ['0001', '0002', '0005',  '0011', '0013', '0014', '0015', '0017', '0018', '0019', '0020', '0022', '0023', '0027', '0028', '0029', '0032', '0035', '0036', '0039', '0046', '0048', '0051', '0052', '0056', '0057', '0059', '0060', '0061']
	data_path = '../../DenseLidarNet_data'
	full_pickle_data = []

	for seq_id in seq_ids:
		print(seq_id)
		full_pickle_data += process_gt_data(seq_id, data_path)

	with open(os.path.join(data_path, 'all_annt_train.pickle'),'wb') as f:
		pkl.dump(full_pickle_data,f,protocol=pkl.HIGHEST_PROTOCOL)





