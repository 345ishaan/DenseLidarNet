from __future__ import print_function

import os
import sys
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from voxelize import voxelize_lidar
import cPickle as pkl
from PIL import Image,ImageDraw


class DenseLidarGen(data.Dataset):

	def __init__(self,annt_file,data_root,transform):
		self.data_root = data_root
		self.annt_file = annt_file
		self.transform = transform
		self.annts = pkl.load(open(annt_file,'rb'))
		self.num_samples = len(self.annts)
		np.random.shuffle(self.annts)

	def __getitem__(self,idx):
		seq_id = self.annts[idx][0]
		fnum_id = self.annts[idx][1]
		bbox_data = self.annts[idx][2:6]
		lidar_path = self.annts[idx][6]
		img_path = os.path.join(self.data_root,str(seq_id),'image_02','data','%.10d'%(idx)+'.png')
		return img_path,bbox_data,lidar_path

	def __len__(self):
		return self.num_samples

	def collate_fn(self,samples):
		'''
		Just collating the lidar data for version 0.0
		'''	
		batch_lidar_data=[]	
		v_w=0.2
		v_h=0.2
		max_l=4
		max_w=2
		num_voxels =None
		num_pts = None
		for img_path,bbox_data,lidar_path in samples:
			lidar_data = pkl.load(open(lidar_path,'rb'))
			voxel_map = voxelize_lidar(lidar_data,v_w=v_w,v_h=v_h,max_l=max_l,max_w=max_w)
			num_voxels = voxel_map.shape[0]
			num_pts =  voxel_map.shape[1]
			batch_lidar_data.append(self.transform(voxel_map))

		inputs = torch.zeros(len(batch_lidar_data),num_voxels,num_pts,7)
		for i in range(len(batch_lidar_data)):
			inputs[i] = batch_lidar_data[i]

		return inputs

if __name__ == '__main__':
	import torchvision
	import uuid
	import cv2
	from ipdb import set_trace as brk

	transform = transforms.Compose([
		transforms.ToTensor()
	])
	dataset = DenseLidarGen('/home/ishan/DenseLidarNet/code/vfe/all_annt_train.pickle','/mnt/cvrr-nas/WorkArea3/WorkArea3_NoBackup/Userarea/ishan/KITTI_data/',transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn)

	
	niters = 0
	max_iters = 10
	dump_path = './dbg'
	
	if not os.path.exists(dump_path):
		os.makedirs(dump_path)
	else:
		map(lambda x: os.unlink(os.path.join(dump_path,x)), os.listdir(dump_path)) 

	while niters < max_iters:

		for lidar_data in dataloader:
			print (lidar_data.size())