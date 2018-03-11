import sys
import os
import torch
import numpy as np

def voxelize_lidar(lidar_data,tf_lidar_data,voxel_id_offset,voxel_map_h,voxel_map_w,num_voxels,max_pts_voxel,v_w,v_h,max_w,max_l):
	
	
	valid_tf_lidar_data = tf_lidar_data[np.where(tf_lidar_data[:,2]<=0.4)[0],:]
	valid_lidar_data =lidar_data[np.where(tf_lidar_data[:,2]<=0.4)[0],:]

	voxel_indices = []
	voxel_features= []
	voxel_mask = []
	for i in range(int(voxel_map_h)):
		x_end = max_l/2 -i*v_h
		x_start = x_end-v_h

		for j in range(int(voxel_map_w)):
			y_end = max_w/2 -j*v_w
			y_start = y_end- v_w
			valid_pts_idx = np.where((valid_tf_lidar_data[:,0]>x_start) & (valid_tf_lidar_data[:,0]<=x_end)\
									& (valid_tf_lidar_data[:,1]>y_start)& (valid_tf_lidar_data[:,1]<=y_end))[0]
			valid_tf_pts = valid_tf_lidar_data[valid_pts_idx,:]
			valid_lidar_pts = valid_lidar_data[valid_pts_idx,:]
			if len(valid_tf_pts) > max_pts_voxel:
				sampled_idx = np.random.choice(len(valid_tf_pts),max_pts_voxel,replace=False)
				valid_tf_pts = valid_tf_pts[sampled_idx,:]
				valid_lidar_pts = valid_lidar_pts[sampled_idx,:]

			voxel_id = i*voxel_map_w + j

			if len(valid_tf_pts):
				voxel_feat = torch.zeros((max_pts_voxel,7))
				voxel_m = torch.zeros((max_pts_voxel,1))

				voxel_m[:len(valid_tf_pts),0] = 1
				voxel_feat[:len(valid_tf_pts),0] = torch.Tensor(valid_tf_pts[:,0]) #x
				voxel_feat[:len(valid_tf_pts),1] = torch.Tensor(valid_tf_pts[:,1]) #y
				voxel_feat[:len(valid_tf_pts),2] = torch.Tensor(valid_tf_pts[:,2]) #z
				voxel_feat[:len(valid_tf_pts),3] = torch.Tensor(valid_lidar_pts[:,3]) #r
				voxel_feat[:len(valid_tf_pts),4] = torch.Tensor(valid_tf_pts[:,0] - np.median(valid_tf_pts[:,0])) #x-med
				voxel_feat[:len(valid_tf_pts),5] = torch.Tensor(valid_tf_pts[:,1] - np.median(valid_tf_pts[:,1])) #y-med
				voxel_feat[:len(valid_tf_pts),6] = torch.Tensor(valid_tf_pts[:,2] - np.median(valid_tf_pts[:,2])) #z-med
				
				voxel_features.append(voxel_feat)
				voxel_indices.append(voxel_id+voxel_id_offset)
				voxel_mask.append(voxel_m)

	if len(voxel_features) == 0:
		return [],[],[]
	return torch.stack(voxel_features),voxel_indices,torch.stack(voxel_mask)

