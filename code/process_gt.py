import cPickle as pkl
import numpy as np
import os
import pandas as pd
from ipdb import set_trace as brk
import h5py


def process_gt_data(seq_id, data_path):

	bbox_path = '../ground_truth/' + seq_id + '_bbox.h5'
	idx_path = '../ground_truth/' + seq_id + '_idx.h5'
	lidar_path = '../ground_truth/' + seq_id + '_lidar.h5'

	hf_bbox = h5py.File(bbox_path,'r')
	hf_idx = h5py.File(idx_path,'r')
	hf_lidar = h5py.File(lidar_path,'r')

	idx_list = list(hf_idx.get('idx'))

	if not os.path.exists(data_path):
		os.makedirs(data_path)


	pickle_data = []
	for idx in idx_list:
		splits = idx[0].split('_')
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

	seq_ids = ['0035', '0001']
	data_path = '../../DenseLidarNet_data'
	full_pickle_data = []

	for seq_id in seq_ids:
		full_pickle_data += process_gt_data(seq_id, data_path)

	with open(os.path.join(data_path, 'all_annt_train.pickle'),'wb') as f:
		pkl.dump(full_pickle_data,f,protocol=pkl.HIGHEST_PROTOCOL)





