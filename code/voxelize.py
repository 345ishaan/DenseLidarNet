import sys
import os
import torch
import numpy as np

def voxelize_lidar(lidar_data,tf_lidar_data,delta_x,delta_z,max_x,min_x,max_y,min_y,max_z,min_z,max_pts_voxel):
	
	valid_tf_lidar_data = tf_lidar_data[np.where(tf_lidar_data[:,1]<=0.4)[0],:]
	valid_lidar_data =lidar_data[np.where(tf_lidar_data[:,1]<=0.4)[0],:]
	voxel_indices = []
	voxel_features= []
	voxel_mask = []
        voxel_map_w = int((max_x - min_x)/float(delta_x)) #x,z in  camera coordinate
	voxel_map_l = int((max_z - min_z)/float(delta_z)) #x,z in camera coordinate
	for i in range(voxel_map_w):
		#Assumption:- max_x = abs(min_x)
		#Assumption:- max_z = abs(min_z)
		x_start = delta_x*i + min_x
		x_end = x_start + delta_x

		for j in range(voxel_map_l):
			z_start = delta_z*j + min_z
			z_end = z_start + delta_z
			valid_pts_idx = np.where((valid_tf_lidar_data[:,0]>=x_start) & (valid_tf_lidar_data[:,0]<x_end)\
						& (valid_tf_lidar_data[:,2]>=z_start) & (valid_tf_lidar_data[:,2]<=z_end))[0]
			valid_tf_pts = valid_tf_lidar_data[valid_pts_idx,:]
			valid_lidar_pts = valid_lidar_data[valid_pts_idx,:]
			#Todo(Ishan) :- Change Sampling Technique
			if len(valid_tf_pts) > max_pts_voxel:
				sampled_idx = np.random.choice(len(valid_tf_pts),max_pts_voxel,replace=False)
				valid_tf_pts = valid_tf_pts[sampled_idx,:]
				valid_lidar_pts = valid_lidar_pts[sampled_idx,:]

			voxel_id = j*voxel_map_w + i
			num_valid_pts = len(valid_tf_pts)
			if num_valid_pts:
				voxel_feat = torch.zeros((max_pts_voxel,7))
				voxel_m = torch.zeros((max_pts_voxel,1))

				voxel_m[:num_valid_pts,0] = 1
				voxel_feat[:num_valid_pts,0] = torch.Tensor(valid_tf_pts[:,0]) #x
				voxel_feat[:num_valid_pts,1] = torch.Tensor(valid_tf_pts[:,1]) #y
				voxel_feat[:num_valid_pts,2] = torch.Tensor(valid_tf_pts[:,2]) #z
				voxel_feat[:num_valid_pts,3] = torch.Tensor(valid_lidar_pts[:,3]) #r
				#Todo(Ishan) Directly use these features with reflectance. Think memory
				voxel_feat[:num_valid_pts,4] = torch.Tensor(valid_tf_pts[:,0] - np.median(valid_tf_pts[:,0])) #x-med
				voxel_feat[:num_valid_pts,5] = torch.Tensor(valid_tf_pts[:,1] - np.median(valid_tf_pts[:,1])) #y-med
				voxel_feat[:num_valid_pts,6] = torch.Tensor(valid_tf_pts[:,2] - np.median(valid_tf_pts[:,2])) #z-med
				
				voxel_features.append(voxel_feat)
				voxel_indices.append(voxel_id)
				voxel_mask.append(voxel_m)

	if len(voxel_features) == 0:
		return [],[],[]
	return torch.stack(voxel_features),torch.LongTensor(voxel_indices),torch.stack(voxel_mask)

