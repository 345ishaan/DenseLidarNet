import sys
import os
import numpy as np

def voxelize_lidar(lidar_data,v_w=0.2,v_h=0.2,max_l=4,max_w=2,max_points=35):
	
	valid_lidar_data =lidar_data[np.where(lidar_data[:,2]<=0.4)[0],:]
	voxel_map_h = max_l/v_h
	voxel_map_w = max_w/v_w
	num_voxels = int(voxel_map_h*voxel_map_w)
	
	voxel_map = np.zeros((num_voxels,max_points,7))
	for i in range(int(voxel_map_h)):
		x_end = max_l/2 -i*v_h
		x_start = x_end-v_h

		for j in range(int(voxel_map_w)):
			y_end = max_w/2 -j*v_w
			y_start = y_end- v_w
			valid_pts_idx = np.where((valid_lidar_data[:,0]>x_start) & (valid_lidar_data[:,0]<=x_end)\
									& (valid_lidar_data[:,1]>y_start)& (valid_lidar_data[:,1]<=y_end))[0]
			valid_pts = valid_lidar_data[valid_pts_idx,:]
			if len(valid_pts) > max_points:
				valid_pts = valid_pts[np.random.choice(len(valid_pts),max_points,replace=False),:]

			voxel_id = i*voxel_map_w + j
			if len(valid_pts):
				voxel_map[voxel_id,:len(valid_pts),0] = valid_pts[:,0] #x
				voxel_map[voxel_id,:len(valid_pts),1] = valid_pts[:,1] #y
				voxel_map[voxel_id,:len(valid_pts),2] = valid_pts[:,2] #z
				voxel_map[voxel_id,:len(valid_pts),3] = valid_pts[:,3] #r

				voxel_map[voxel_id,:len(valid_pts),4] = valid_pts[:,0] - np.median(valid_pts[:,0]) #x-med
				voxel_map[voxel_id,:len(valid_pts),5] = valid_pts[:,1] - np.median(valid_pts[:,1]) #y-med
				voxel_map[voxel_id,:len(valid_pts),6] = valid_pts[:,2] - np.median(valid_pts[:,2]) #z-med
	return voxel_map

