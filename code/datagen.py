import  sys
import os
import glob
from PIL import Image, ImageDraw,ImageFont
import cv2
from parseTrackletXML import *


class DataLoader(object):

	def __init__(self,data_root,seq_id):
		self.data_root = data_root
		self.seq_id = str(seq_id)
		self.lidar_root = os.path.join(self.data_root,self.seq_id,'velodyne_points','data')
		self.image_root = os.path.join(self.data_root,self.seq_id,'image_02','data')
		
		self.image_paths = sorted(glob.glob(os.path.join(self.image_root,'*.png')))
		self.lidar_paths = sorted(glob.glob(os.path.join(self.lidar_root,'*.bin')))

		assert len(self.image_paths) == len(self.lidar_paths)

		self.bev_resolution=[0.1,0.1]
		self.bev_x_range=[-70.0,70.0]
		self.bev_y_range=[-30.0,30.0]

		self.counter =0

	def get_lidar_pts(self):
		data = np.fromfile(self.lidar_paths[self.counter], np.float32)
		data = data.reshape(data.shape[0] // 4, 4)
		return data

	def get_image_data(self):
		data = cv2.imread(self.image_paths[self.counter])
		return data

	def gen_bird_view(self,pts):
		min_x = self.bev_x_range[0]
		max_x = self.bev_x_range[1]

		min_y = self.bev_y_range[0]
		max_y = self.bev_y_range[1]

		height = int((max_x - min_x)/self.bev_resolution[0])
		width = int((max_y - min_y)/self.bev_resolution[0])

		valid_idx = np.where((pts[:,0] > min_x) & (pts[:,0] < max_x) & (pts[:,1] > min_y) & (pts[:,1] < max_y))[0]
		valid_pts = pts[valid_idx,:]
		
		valid_pts = valid_pts[:,:2]
		valid_pts = valid_pts - np.asarray([min_x,min_y]).reshape(-1,2)
		valid_pts = valid_pts / np.asarray([max_x-min_x,max_y-min_y]).reshape(-1,2)
		valid_pts = valid_pts * np.asarray([height,width]).reshape(-1,2)
		valid_pts = valid_pts.astype(np.int32)

		bird_view = np.zeros((height,width,3))
		
		bird_view[valid_pts[:,0],valid_pts[:,1],:] =1
		return bird_view

	def process_data(self):

		while(self.counter < len(self.image_paths)):
			lidar_data = self.get_lidar_pts()
			img_bgr = self.get_image_data()
			bev = self.gen_bird_view(lidar_data)
			composite = np.zeros((bev.shape[0],bev.shape[1]+img_bgr.shape[1],3))
			composite[:,:bev.shape[1],:] = bev*255
			composite[350:350+img_bgr.shape[0],bev.shape[1]:,:] = img_bgr
			cv2.imwrite('composite.png',composite)
			self.counter += 1
			break


if __name__ == '__main__':
	data_root = '../data/KITTI_data'
	loader = DataLoader(data_root,1)
	loader.process_data()




