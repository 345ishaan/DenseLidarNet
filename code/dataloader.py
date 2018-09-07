from __future__ import print_function

import os
import sys
import glob
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from voxelize import voxelize_lidar
from ipdb import set_trace as brk
if sys.version_info >= (3,0):
	import _pickle as pkl
else:
	import cPickle as pkl
from PIL import Image,ImageDraw


class DenseLidarGen(data.Dataset):

	def __init__(self,lidar_pts_path,tf_lidar_pts_path,bbox_info_path,transform):
		self.lidar_pts_path = lidar_pts_path
		self.tf_lidar_pts_path = tf_lidar_pts_path
		self.bbox_info_path = bbox_info_path
		self.transform = transform

		self.tf_lidar_pts_files = glob.glob(os.path.join(self.tf_lidar_pts_path,"*.npy"))
		self.lidar_pts_files = glob.glob(os.path.join(self.lidar_pts_path,"*.npy"))
		self.bbox_info_files = glob.glob(os.path.join(self.bbox_info_path,"*.txt"))
		self.num_samples = len(self.tf_lidar_pts_files)
		self.dx = 0.2
		self.dz = 0.2
		#Assuming max width of vehicle to be 2
		self.max_x = 1
		self.min_x = -1
		#Assuming max length of vehicle to be 4
		self.max_z = 2
		self.min_z = -2

		self.max_y = 0.4
		self.min_y = -3

		self.max_pts_in_voxel = 35
		
		#This number can not go bigger than the threshold set while dumping data
		self.max_pred_pts = 500

	def __getitem__(self,idx):
		fname = self.tf_lidar_pts_files[idx].split('/')[-1]
		tf_lidar_pts = np.load(self.tf_lidar_pts_files[idx])
		lidar_pts = np.load(os.path.join(self.lidar_pts_path,fname))
		voxel_features,voxel_indices,voxel_mask = voxelize_lidar(lidar_pts,tf_lidar_pts,\
			self.dx,self.dz,self.max_x,self.min_x,self.max_y,self.min_y,self.max_z,self.min_z,self.max_pts_in_voxel)
		gt_pts = tf_lidar_pts[np.random.choice(tf_lidar_pts.shape[0],self.max_pred_pts,replace=False),:3]
		return voxel_features,voxel_indices,voxel_mask,gt_pts
	

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
		
		for voxel_features,voxel_indices,voxel_mask,gt_pts in samples:
			if not len(voxel_features):
				continue
			batch_voxel_data.append(voxel_features)
			batch_voxel_mask.append(voxel_mask)
			batch_voxel_indices.append(voxel_indices)
			batch_chamfer_gt.append(gt_pts)
		
		if len(batch_voxel_data) == 0:
			return [],[],[],[]
		
		return batch_voxel_data,batch_voxel_mask,batch_voxel_indices,batch_chamfer_gt

if __name__ == '__main__':
	import torchvision
	import uuid
	import cv2
	from ipdb import set_trace as brk

	transform = transforms.Compose([
		transforms.ToTensor()
	])
	lidar_pts_path = "/tmp/DenseLidarNet/lidar_pts"
	tf_lidar_pts_path = "/tmp/tf_lidar_pts"
	bbox_info_path = "/tmp/DenseLidarNet/bbox_info"

	dataset = DenseLidarGen(lidar_pts_path,tf_lidar_pts_path,bbox_info_path,transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn)
		
	for voxel_data,voxel_mask,voxel_indices,voxel_gt in dataloader:
		print (voxel_data[0].shape)
		print (voxel_mask[0].shape)
		print (voxel_indices[0].shape)
		print (voxel_gt[0].shape)
