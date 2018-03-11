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
		tf_lidar_path = self.annts[idx][7]
		img_path = os.path.join(self.data_root,str(seq_id),'image_02','data','%.10d'%(idx)+'.png')
		return img_path,bbox_data,lidar_path,tf_lidar_path

	def __len__(self):
		return self.num_samples

	def collate_fn(self,samples):
		'''
		Just collating the lidar data for version 0.0
		'''	
		batch_voxel_data=[]	
		batch_voxel_mask=[]	
		batch_voxel_indices=[]
		batch_chamfer_gt = []


		v_w=0.2
		v_h=0.2
		max_l=4
		max_w=2
		voxel_map_h = int(max_l/v_h)
		voxel_map_w = int(max_w/v_w)
		voxel_id_offset = 0
		num_voxels =voxel_map_w*voxel_map_h
		max_pts_in_voxel = 35
		
		
		for img_path,bbox_data,lidar_path,tf_lidar_path in samples:
			lidar_data = pkl.load(open(lidar_path,'rb'))
			tf_lidar_data = pkl.load(open(tf_lidar_path,'rb'))

			voxel_features,voxel_indices,voxel_mask = voxelize_lidar(lidar_data,tf_lidar_data,voxel_id_offset,voxel_map_h,voxel_map_w,num_voxels,max_pts_in_voxel,v_w,v_h,max_w,max_l)
			if type(voxel_features) != list:
				voxel_id_offset += num_voxels
				batch_voxel_data.append(voxel_features)
				batch_voxel_mask.append(voxel_mask)
				batch_voxel_indices.extend(voxel_indices)

				the_pts = tf_lidar_data[np.random.choice(tf_lidar_data.shape[0],1000,replace=True),:3] # change to F
				batch_chamfer_gt.append(torch.FloatTensor(the_pts).unsqueeze(0))

		
		if len(batch_voxel_data) == 0:
			return [],[],[],[]
		
		return torch.cat(batch_voxel_data,0),torch.cat(batch_voxel_mask,0),\
				torch.LongTensor((batch_voxel_indices)),torch.cat(batch_chamfer_gt,0)

if __name__ == '__main__':
	import torchvision
	import uuid
	import cv2
	from ipdb import set_trace as brk

	transform = transforms.Compose([
		transforms.ToTensor()
	])
	dataset = DenseLidarGen('../../DenseLidarNet_data/all_annt_train.pickle','./imgs',transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn)

	
	niters = 0
	max_iters = 1000000
	dump_path = './dbg'
	
	if not os.path.exists(dump_path):
		os.makedirs(dump_path)
	else:
		map(lambda x: os.unlink(os.path.join(dump_path,x)), os.listdir(dump_path)) 

	while niters < max_iters:

		for voxel_features,voxel_mask,voxel_indices,gt in dataloader:
			if(type(voxel_features) == list):
				continue

			print (voxel_features.size())
			print (voxel_mask.size())
			print (voxel_indices.size())
			print (gt.size())
			niters += 1

			if niters >max_iters:
				break