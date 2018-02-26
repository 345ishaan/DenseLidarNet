import  sys
import os
import glob
from PIL import Image, ImageDraw,ImageFont
import cv2
from parseTrackletXML import *
from ipdb import set_trace as brk
import pickle


class DataLoader(object):

	def __init__(self,data_root,seq_id):
		self.data_root = data_root
		self.seq_id = str(seq_id)
		self.lidar_root = os.path.join(self.data_root,self.seq_id,'velodyne_points','data')
		self.image_root = os.path.join(self.data_root,self.seq_id,'image_02','data')
		self.tracklet_path = os.path.join(self.data_root,self.seq_id,'tracklet_labels.xml')

		self.image_paths = sorted(glob.glob(os.path.join(self.image_root,'*.png')))
		self.lidar_paths = sorted(glob.glob(os.path.join(self.lidar_root,'*.bin')))

		assert len(self.image_paths) == len(self.lidar_paths)

		self.bev_resolution=[0.1,0.1]
		self.bev_x_range=[-70.0,70.0]
		self.bev_y_range=[-30.0,30.0]

		dir = os.path.dirname(__file__)
		self.video_dir_path = os.path.join(dir, '../videos/')
		self.gt_lidar_path = os.path.join(dir, '../pickles/')

		self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self.bev_video = cv2.VideoWriter(self.video_dir_path + 'bev_video_seq_'+ str(self.seq_id) + '.avi',self.fourcc,30.0,(600, 1400))
		self.res_3d_video = cv2.VideoWriter(self.video_dir_path + 'res_3d_video_seq_'+ str(self.seq_id) + '.avi',self.fourcc,30.0,(1242, 375))


		self.counter =0
		self.num_frames = len(self.image_paths)
		self.objects = ['Car','Van','Truck']
		self.load_projection_matrices()

		self.filtered_lidar_pts = []
	

	def load_projection_matrices(self):
		self.camera_mat_dict ={}
		self.camera_mat_dict['Tr_velo_to_cam'] = np.asarray([[7.533745e-03,-9.999714e-01 ,-6.166020e-04,-4.069766e-03],[ 1.480249e-02 , 7.280733e-04,  -9.998902e-01, -7.631618e-02],[9.998621e-01, 7.523790e-03 ,1.480755e-02,-2.717806e-01],[0.0, 0.0, 0.0, 1.0]])
		self.camera_mat_dict['P_Rectified'] = np.asarray([[7.215377e+02, 0.000000e+00,6.095593e+02 ,4.485728e+01],[0.000000e+00 ,7.215377e+02 ,1.728540e+02 ,2.163791e-01],[ 0.000000e+00 ,0.000000e+00, 1.000000e+00, 2.745884e-03]])    
		self.camera_mat_dict['R_Rectified'] = np.asarray([[9.999239e-01 ,9.837760e-03 ,-7.445048e-03, 0.00],[-9.869795e-03 , 9.999421e-01 , -4.278459e-03, 0.00],[7.402527e-03, 4.351614e-03, 9.999631e-01, 0.00],[0.0, 0.0, 0.0, 1.0]])
		
	def project_to_image(self,lidar_pts):
		homo_lidar_pts = np.concatenate([lidar_pts,np.ones((1,lidar_pts.shape[1]))],0)
		homo_camera_coord_pts = self.camera_mat_dict['Tr_velo_to_cam'].dot(homo_lidar_pts)
		homo_image_plane_pts =  self.camera_mat_dict['P_Rectified'].dot(self.camera_mat_dict['R_Rectified'].dot(homo_camera_coord_pts))
		homo_image_plane_pts = homo_image_plane_pts/homo_image_plane_pts[-1,:]
		return homo_image_plane_pts[:2,:]

	def get_lidar_pts(self,index):
		
		data = np.fromfile(self.lidar_paths[index], np.float32)
		data = data.reshape(data.shape[0] // 4, 4)
		return data

	def get_image_data(self,index):
		data = cv2.imread(self.image_paths[index])
		return data

	def gen_bird_view(self,pts=None):
		min_x = self.bev_x_range[0]
		max_x = self.bev_x_range[1]

		min_y = self.bev_y_range[0]
		max_y = self.bev_y_range[1]

		height = int((max_x - min_x)/self.bev_resolution[0])
		width = int((max_y - min_y)/self.bev_resolution[0])
		bird_view = np.zeros((height,width,3))
		if pts is not None:
			valid_idx = np.where((pts[:,0] > min_x) & (pts[:,0] < max_x) & (pts[:,1] > min_y) & (pts[:,1] < max_y))[0]
			valid_pts = pts[valid_idx,:]
			
			valid_pts = valid_pts[:,:2]
			valid_pts = valid_pts - np.asarray([min_x,min_y]).reshape(-1,2)
			valid_pts = valid_pts / np.asarray([max_x-min_x,max_y-min_y]).reshape(-1,2)
			valid_pts = valid_pts * np.asarray([height,width]).reshape(-1,2)
			valid_pts = valid_pts.astype(np.int32)

		
			bird_view[valid_pts[:,0],valid_pts[:,1],:] =1

		return bird_view

	def get_all_tracklets(self):
		self.tracklet_data = parse_XML(self.tracklet_path,self.num_frames,object_of_interest=self.objects)

	def get_tracklet_pts(self,lidar_pts,rot_mat,center,dims):
		'''
		Note
		The ground plane is not straight, hence we can't keep the z range check
			Need to clarify 
		'''

		l,w,h = dims
		
		tf_pts = (center.reshape(-1,3) - lidar_pts).dot(rot_mat)
		
		valid_pts = np.where(#(tf_pts[:,2] <= h) & (tf_pts[:,2] >= 0) &\
							 (tf_pts[:,1] <= w/2) & (tf_pts[:,1] >= -w/2) &\
							 (tf_pts[:,0] <= l/2) & (tf_pts[:,0] >= -l/2) )[0]

		return lidar_pts[valid_pts,:]

	def get_2d_bbox(self,pts):
		min_x = np.min(pts[0,:])
		min_y = np.min(pts[1,:])
		max_x = np.max(pts[0,:])
		max_y = np.max(pts[1,:])
		return (min_x,min_y,max_x,max_y)

	def draw_3d_box(self,points,img,color=(255,255,0)):
		points = points.astype(np.int32)
		
		cv2.line(img,tuple(points[:,0]),tuple(points[:,1]),color,2)
		cv2.line(img,tuple(points[:,1]),tuple(points[:,2]),color,2)
		cv2.line(img,tuple(points[:,2]),tuple(points[:,3]),color,2)
		cv2.line(img,tuple(points[:,0]),tuple(points[:,3]),color,2)

		cv2.line(img,tuple(points[:,4]),tuple(points[:,5]),color,2)
		cv2.line(img,tuple(points[:,5]),tuple(points[:,6]),color,2)
		cv2.line(img,tuple(points[:,6]),tuple(points[:,7]),color,2)
		cv2.line(img,tuple(points[:,4]),tuple(points[:,7]),color,2)

		cv2.line(img,tuple(points[:,0]),tuple(points[:,4]),color,2)
		cv2.line(img,tuple(points[:,1]),tuple(points[:,5]),color,2)
		cv2.line(img,tuple(points[:,2]),tuple(points[:,6]),color,2)
		cv2.line(img,tuple(points[:,3]),tuple(points[:,7]),color,2)


	def draw_rectangle_cv(self,img,box,mode='xyxy',color=(0,255,0)):
		if mode == 'xyxy':
			cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color,2)
		else:
			cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2])+int(box[0]),int(box[3])+int(box[1])),color,2)
		
	def process_data(self):

		while(self.counter < len(self.image_paths)):
			lidar_data = self.get_lidar_pts(self.counter)
			img_bgr = self.get_image_data(self.counter)
			bev = self.gen_bird_view(lidar_data)
			composite = np.zeros((bev.shape[0],bev.shape[1]+img_bgr.shape[1],3))
			composite[:,:bev.shape[1],:] = bev*255
			composite[350:350+img_bgr.shape[0],bev.shape[1]:,:] = img_bgr
			cv2.imwrite('composite.png',composite)
			self.counter += 1
			break
	
	def get_all_annt(self):
		
		self.get_all_tracklets()
		for i in range(self.num_frames):
			frm_data = self.tracklet_data[i]
			img_bgr = self.get_image_data(i)
			all_lidar_pts = self.get_lidar_pts(i)
			all_lidar_pts = all_lidar_pts[:,:3]

			global_bev = self.gen_bird_view()

			for j in range(len(frm_data)):
				lidar_pts = frm_data[j][:,:8]
				center = frm_data[j][:,8]
				yaw = np.unique(frm_data[j][:,10])[0]
				dim = (frm_data[j][0,[-6,-5,-4]]).tolist()
				rot_mat = np.asarray([[np.cos(yaw), np.sin(yaw), 0.0],[-np.sin(yaw), np.cos(yaw), 0.0],[0.0,0.0, 1.0]])
				pts_in_image = self.project_to_image(lidar_pts)
				bbox = self.get_2d_bbox(pts_in_image)
				
				cor_pts = self.get_tracklet_pts(all_lidar_pts,rot_mat,center,dim)
				
				
				bird_view = self.gen_bird_view(cor_pts)
				global_bev = global_bev + bird_view
				self.draw_3d_box(pts_in_image,img_bgr)
				# self.draw_rectangle_cv(img_bgr,bbox)
			# cv2.imwrite('bird_view_{}_tushar.png'.format(i),global_bev*255)
			# cv2.imwrite('res_3d_vis_{}_tushar.png'.format(i),img_bgr)
			self.filtered_lidar_pts.append(cor_pts)
			self.bev_video.write(np.array(global_bev*255, dtype=np.uint8))
			self.res_3d_video.write(img_bgr)
		self.bev_video.release()
		self.res_3d_video.release()
		with open(self.gt_lidar_path + str(self.seq_id) + '.pkl', 'wb') as f:
			pickle.dump(self.filtered_lidar_pts, f)


if __name__ == '__main__':
# 	data_root = '/home/ishan/Downloads/KITTI_DATA'
	data_root = '../KITTI_data'
	seq_id = 35
	loader = DataLoader(data_root,seq_id)
	loader.get_all_annt()




