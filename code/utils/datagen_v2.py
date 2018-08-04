import  sys
import os
import glob
from math import sin,cos
import cPickle as pkl
from tqdm import tqdm
# from PIL import Image, ImageDraw,ImageFont
import cv2
from parseTrackletXML import *
from ipdb import set_trace as brk
import h5py

class DataLoader(object):

	def __init__(self,train_val_split=0.2):
		
		self.kitti_img_dir = '../../data/images/training/image_2/'
		self.kitti_calib_dir = '../../data/calibration/training/calib/'
		self.kitti_label_dir = '../../data/labels/label_2/'
		self.kitti_lidar_dir = '../../data/lidar/training/velodyne'
		
		self.model_op_files = sorted(glob.glob('./dump_3d_data/kitti_train/14/*.txt'))
		self.kitti_label_files = sorted(glob.glob(self.kitti_label_dir+'/*.txt'))
		np.random.shuffle(self.kitti_label_files)

		self.train_label_files = self.kitti_label_files[:int(len(self.kitti_label_files)*(1-train_val_split))]
		self.val_label_files = self.kitti_label_files[int(len(self.kitti_label_files)*(1-train_val_split)):]
		
		self.num_train_files = len(self.train_label_files)
		self.num_val_files = len(self.val_label_files)

		print ("{} Train Files, {} Validation Files".format(self.num_train_files,self.num_val_files))

		self.bev_resolution=[0.1,0.1]
		self.bev_x_range=[-70.0,70.0]
		self.bev_y_range=[-30.0,30.0]

		self.dump_dir = '../../data/'


		
		#self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
		#self.bev_video = cv2.VideoWriter(self.video_dir_path + 'bev_video_seq_'+ self.seq_id + '.avi',self.fourcc,30.0,(600, 1400))
		#self.res_3d_video = cv2.VideoWriter(self.video_dir_path + 'res_3d_video_seq_'+ self.seq_id + '.avi',self.fourcc,30.0,(1242, 375))


		self.counter =0
		self.max_iters = 100#sys.maxint
		
		self.objects = ['Car','Van','Truck']
		
		self.filtered_lidar_pts = []
		self.count_valid_objects = 0

	def read_kitti_labels(self,index):
		f = open(index,'rb')
		lines = f.readlines()
		obj = {'xmin':[],'ymin':[],'xmax':[],'ymax':[],'l':[],'w':[],'h':[],'loc_x':[],'loc_y':[],'loc_z':[],'rot_camera':[],'yaw':[]}
		for line in lines:
			otype,truncated,occluded,visual_yaw,x1,y1,x2,y2,h,w,l,loc_x,loc_y,loc_z,rotation_y= line.split()
			if otype not in self.objects:
				continue
			if float(truncated) > 0.5: # only allow objects with truncation less than 0.5
				continue
			if int(occluded) >=2 : # only allow fully visible, and partially occluded objects
				continue
			rot_camera = float(visual_yaw) + np.arctan(float(loc_x)/float(loc_z))
			obj['xmin'].append(float(x1))
			obj['ymin'].append(float(y1))
			obj['xmax'].append(float(x2))
			obj['ymax'].append(float(y2))

			obj['l'].append(float(l))
			obj['w'].append(float(w))
			obj['h'].append(float(h))

			obj['loc_x'].append(float(loc_x))
			obj['loc_y'].append(float(loc_y))
			obj['loc_z'].append(float(loc_z))

			obj['rot_camera'].append(float(rot_camera))
			obj['yaw'].append(float(rotation_y))

		return obj

	def project_to_img_plane(self,pts,P):
		homo_pts = np.concatenate([pts,np.ones((1,pts.shape[1]))],0)
		proj_pts = P.dot(homo_pts)
		proj_pts = proj_pts/proj_pts[-1,:]
		return proj_pts[:2,:]


	def compute_3d_vertices(self,yaw,translation,dims,P):
		rot_mat =[[cos(yaw),0,sin(yaw)],[0,1,0],[-sin(yaw),0,cos(yaw)]]
		l, w,h = dims[0],dims[1],dims[2]
		vertices = [[l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2,0],\
					[0,0,0,0,-h,-h,-h,-h,0],\
					[w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2,0]]
		rot_vertices = np.asarray(rot_mat).dot(vertices)
		trans_rot_vertices = rot_vertices + np.tile(translation, (9,1)).T

		vertices_in_img = self.project_to_img_plane(trans_rot_vertices,P)
		return vertices_in_img

	def draw_3d_box(self,img,vertices,color=(255,255,0)):
		cv2.line(img,tuple(vertices[:,0]),tuple(vertices[:,1]),color,2)
		cv2.line(img,tuple(vertices[:,1]),tuple(vertices[:,2]),color,2)
		cv2.line(img,tuple(vertices[:,2]),tuple(vertices[:,3]),color,2)
		cv2.line(img,tuple(vertices[:,0]),tuple(vertices[:,3]),color,2)

		cv2.line(img,tuple(vertices[:,4]),tuple(vertices[:,5]),color,2)
		cv2.line(img,tuple(vertices[:,5]),tuple(vertices[:,6]),color,2)
		cv2.line(img,tuple(vertices[:,6]),tuple(vertices[:,7]),color,2)
		cv2.line(img,tuple(vertices[:,4]),tuple(vertices[:,7]),color,2)

		cv2.line(img,tuple(vertices[:,0]),tuple(vertices[:,4]),color,2)
		cv2.line(img,tuple(vertices[:,1]),tuple(vertices[:,5]),color,2)
		cv2.line(img,tuple(vertices[:,2]),tuple(vertices[:,6]),color,2)
		cv2.line(img,tuple(vertices[:,3]),tuple(vertices[:,7]),color,2)


	
	def get_lidar_pts(self,fpath):
		data = np.fromfile(fpath, np.float32)
		data = data.reshape(data.shape[0] // 4, 4)
		return data

	def get_image_data(self,index):
		data = cv2.imread(self.image_paths[index])
		return data

	def gen_bird_view(self,pts=None,use_bev=False,cur_bev=None):
		min_x = self.bev_x_range[0]
		max_x = self.bev_x_range[1]

		min_y = self.bev_y_range[0]
		max_y = self.bev_y_range[1]

		height = int((max_x - min_x)/self.bev_resolution[0])
		width = int((max_y - min_y)/self.bev_resolution[0])
		if not use_bev:
			bird_view = np.zeros((height,width,3))
		else:
			bird_view = cur_bev

		if pts is not None:
			# valid_idx = np.where((pts[:,0] > min_x) & (pts[:,0] < max_x) & (pts[:,1] > min_y) & (pts[:,1] < max_y))[0]
			valid_idx = np.where((pts[:,2] > min_x) & (pts[:,2] < max_x) & (pts[:,0] > min_y) & (pts[:,0] < max_y))[0]
			valid_pts = pts[valid_idx,:]
			
			valid_pts = valid_pts[:,[2,0]]
			valid_pts = valid_pts - np.asarray([min_x,min_y]).reshape(-1,2)
			valid_pts = valid_pts / np.asarray([max_x-min_x,max_y-min_y]).reshape(-1,2)
			valid_pts = valid_pts * np.asarray([height,width]).reshape(-1,2)
			valid_pts = valid_pts.astype(np.int32)

			if not use_bev:
				bird_view[valid_pts[:,0],valid_pts[:,1],:] =255
			else:
				bird_view[valid_pts[:,0],valid_pts[:,1],0] =255
				bird_view[valid_pts[:,0],valid_pts[:,1],1] =255
				bird_view[valid_pts[:,0],valid_pts[:,1],2] =0

		return bird_view

	# def get_all_tracklets(self):
	# 	self.tracklet_data = parse_XML(self.tracklet_path,self.num_frames,object_of_interest=self.objects)

	# def get_tracklet_pts(self,lidar_pts,rot_mat,center,dims):
	# 	'''
	# 	Note
	# 	The ground plane is not straight, hence we can't keep the z range check
	# 		Need to clarify 
	# 	'''

	# 	l,w,h = dims
	# 	# brk()
	# 	# tf_pts = (np.asarray(rot_mat).T.dot((lidar_pts[:,:3] - center.reshape(-1,3)).T)).T
	# 	diff = (lidar_pts[:,:3] - center.reshape(-1,3)).T
	# 	tf_pts = np.asarray(rot_mat).dot(diff)
	# 	tf_pts = tf_pts.T
	# 	# np.hstack((tf_pts, lidar_pts[:,3:]))

	# 	valid_pts = np.where(#(tf_pts[:,1] >= -h) & (tf_pts[:,1] <= 0) &\
	# 						 (tf_pts[:,0] <= w/2) & (tf_pts[:,0] >= -w/2) &\
	# 						 (tf_pts[:,2] <= l/2) & (tf_pts[:,2] >= -l/2) )[0]

		

	# 	return {'lidar_pts' : lidar_pts[valid_pts,:], 'tf_pts': tf_pts[valid_pts,:], 'center': center, 'dims': dims}

	def get_all_tracklets(self):
		self.tracklet_data = parse_XML(self.tracklet_path,self.num_frames,object_of_interest=self.objects)

	def get_tracklet_pts(self,lidar_pts,x_cords,z_cords):
		'''
		Note
		The ground plane is not straight, hence we can't keep the z range check
			Need to clarify 
		'''

		min_x = np.min(x_cords)
		max_x = np.max(x_cords)

		min_z = np.min(z_cords)
		max_z = np.max(z_cords)

		valid_pts = np.where((lidar_pts[:,0] <= max_x) & (lidar_pts[:,0] >= min_x) &\
							 (lidar_pts[:,2] <= max_z) & (lidar_pts[:,2] >= min_z) )[0]

		return {'lidar_pts' : lidar_pts[valid_pts,:]}
	
	def get_2d_bbox(self,pts):
		min_x = np.min(pts[0,:])
		min_y = np.min(pts[1,:])
		max_x = np.max(pts[0,:])
		max_y = np.max(pts[1,:])
		return (min_x,min_y,max_x,max_y)

	
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

	def check_path(self,path):
		if not os.path.exists(path):
			os.makedirs(path)
		else:
			map(lambda x :os.unlink(os.path.join(path,x)), os.listdir(path))

	def get_all_annts(self):
		
		self.dump_path = os.path.join(self.dump_dir,'dump');self.check_path(self.dump_path)
		self.tf_pts_dump_path = os.path.join(self.dump_dir,'tf_lidar_pts');self.check_path(self.tf_pts_dump_path)
		self.lidar_pts_dump_path = os.path.join(self.dump_dir,'lidar_pts');self.check_path(self.lidar_pts_dump_path)
		self.meta_info = os.path.join(self.dump_dir,'meta_info');self.check_path(self.meta_info);self.check_path(self.meta_info)

		for it,file in tqdm(enumerate(self.train_label_files)):
			file_id = file.split('/')[-1].split('.')[0]
			
			img_fname = self.kitti_img_dir+file.split('/')[-1].replace('txt','png')
			
			lidar_fname = os.path.join(self.kitti_lidar_dir,file.split('/')[-1].replace('txt','bin'))
			img_bgr = cv2.imread(img_fname)

			calib_fp = open(self.kitti_calib_dir+file.split('/')[-1],'rb')
			calib_fp.readline()
			calib_fp.readline()
			P = calib_fp.readline().strip().split(' ')[1:]
			P = np.asarray(P).astype(np.float32).reshape(3,4)
			
			calib_fp.readline()
			r0_rectified = calib_fp.readline().strip().split(' ')[1:]
			r0_rectified = np.asarray(r0_rectified).astype(np.float32).reshape(3,3)
			r0_rectified = np.concatenate([r0_rectified,np.asarray([0,0,0]).reshape(3,1)],axis=1)
			r0_rectified = np.concatenate([r0_rectified,np.asarray([0,0,0,1]).reshape(-1,4)],axis=0)
			
			velo_to_cam = calib_fp.readline().strip().split(' ')[1:]
			velo_to_cam = np.asarray(velo_to_cam).astype(np.float32).reshape(3,4)
			velo_to_cam = np.concatenate([velo_to_cam,np.asarray([0,0,0,1]).reshape(-1,4)],axis=0)

			label_info = self.read_kitti_labels(file)
			all_lidar_pts = self.get_lidar_pts(lidar_fname)

			velo_xyz = all_lidar_pts[:,:-1]
			velo_ref = all_lidar_pts[:,-1]
			velo_xyz = np.concatenate([velo_xyz.T,np.ones((1,velo_xyz.shape[0]))],axis=0)
			
			velo_to_cam_xyz = ((velo_to_cam.dot(velo_xyz)))
			
			velo_to_cam_xyz = velo_to_cam_xyz/velo_to_cam_xyz[-1,:]
			
			all_lidar_pts_camera = np.concatenate([velo_to_cam_xyz[:-1,:].T,velo_ref.reshape(-1,1)],axis=1)
			
			bird_view = self.gen_bird_view(pts=all_lidar_pts_camera)

			# bird_view[:,:,:] = 0

			for i in range(len(label_info['loc_x'])):
				translation = np.asarray([label_info['loc_x'][i],label_info['loc_y'][i],label_info['loc_z'][i]])
				rot_mat =[[cos(label_info['yaw'][i]),0,sin(label_info['yaw'][i])],[0,1,0],[-sin(label_info['yaw'][i]),0,cos(label_info['yaw'][i])]]
				rot_mat_t =[[cos(label_info['yaw'][i]),0,-sin(label_info['yaw'][i])],[0,1,0],[sin(label_info['yaw'][i]),0,cos(label_info['yaw'][i])]]

				
				# if lidar_dict['tf_pts'].shape[0] < 1000:
				# 	continue

				# print ("Number of Lidar Pts : {}".format(lidar_dict['tf_pts'].shape))
				# bird_view = self.gen_bird_view(pts=lidar_dict['lidar_pts'],use_bev=True,cur_bev=bird_view)
				
				file_save_name = str(file_id) + '_' + str(i)
				
				# pkl.dump(lidar_dict['tf_pts'],open(os.path.join(self.tf_pts_dump_path,file_save_name+'_.pkl'),'wb'))
				# pkl.dump(lidar_dict['lidar_pts'],open(os.path.join(self.lidar_pts_dump_path,file_save_name+'_.pkl'),'wb'))

				meta_fp = open(os.path.join(self.meta_info,file_save_name+'_.txt'),'wb')
				meta_fp.write('{},{},{}'.format(img_fname,label_info['xmin'][i],label_info['ymin'][i],label_info['xmax'][i],label_info['ymax'][i]))
				meta_fp.close()

				
				l, w,h = label_info['l'][i],label_info['w'][i],label_info['h'][i]


				vertices = [[l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2,0],\
							[0,0,0,0,-h,-h,-h,-h,0],\
							[w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2,0]]

				rot_vertices = np.asarray(rot_mat).dot(vertices)
				trans_rot_vertices = rot_vertices + np.tile(translation, (9,1)).T
				

				'''verify'''
				recover_coords = np.asarray(rot_mat).T.dot(trans_rot_vertices - np.tile(translation.reshape(-1,3),(9,1)).T)
				x_cords = trans_rot_vertices[0,:]
				z_cords = trans_rot_vertices[2,:]

				# min_x = (np.min(x_cords) - self.bev_y_range[0])/(self.bev_resolution[0])
				# max_x = (np.max(x_cords) - self.bev_y_range[0])/(self.bev_resolution[0])
				# min_z = (np.min(z_cords) - self.bev_x_range[0])/(self.bev_resolution[1])
				# max_z = (np.max(z_cords) - self.bev_x_range[0])/(self.bev_resolution[1])
				
				# self.draw_rectangle_cv(bird_view,(min_x,min_z,max_x,max_z))
				lidar_dict = self.get_tracklet_pts(all_lidar_pts_camera,x_cords,z_cords)
				if lidar_dict['lidar_pts'].shape[0] >= 500:
					self.count_valid_objects += 1
					color = (0,255,0)
				else:
					color = (0,0,255)
				bird_view = self.gen_bird_view(pts=lidar_dict['lidar_pts'],use_bev=True,cur_bev=bird_view)
				x1 =  int((x_cords[0] - self.bev_y_range[0])/(self.bev_resolution[0]))
				x2 =  int((x_cords[1] - self.bev_y_range[0])/(self.bev_resolution[0]))
				x3 =  int((x_cords[2] - self.bev_y_range[0])/(self.bev_resolution[0]))
				x4 =  int((x_cords[3] - self.bev_y_range[0])/(self.bev_resolution[0]))

				y1 =  int((z_cords[0] - self.bev_x_range[0])/(self.bev_resolution[0]))
				y2 =  int((z_cords[1] - self.bev_x_range[0])/(self.bev_resolution[0]))
				y3 =  int((z_cords[2] - self.bev_x_range[0])/(self.bev_resolution[0]))
				y4 =  int((z_cords[3] - self.bev_x_range[0])/(self.bev_resolution[0]))

				# color = (0,255,0)
				cv2.line(bird_view,(x1,y1),(x2,y2),color,2)
				cv2.line(bird_view,(x2,y2),(x3,y3),color,2)
				cv2.line(bird_view,(x3,y3),(x4,y4),color,2)
				cv2.line(bird_view,(x4,y4),(x1,y1),color,2)

				
				# paint_indices = np.where()
			
				pts_in_2d = self.compute_3d_vertices(label_info['yaw'][i],translation,(label_info['l'][i],label_info['w'][i],label_info['h'][i]),P)
				
				self.draw_3d_box(img_bgr,pts_in_2d.astype(np.int32))

			cv2.imwrite(os.path.join(self.dump_path,'bev_{}.png').format(file_id),bird_view)
			cv2.imwrite(os.path.join(self.dump_path,'bgr_{}.png').format(file_id),img_bgr)
			
			if it > self.max_iters:
				break
		print "Found {} Valid Objects".format(self.count_valid_objects)
			
			
			
			

	def get_all_annt(self):

		# gt_seq_id_path = self.gt_path + str(self.seq_id)
		
		# try:
  #       	os.makedirs(gt_seq_id_path)
  #   	except OSError as exception:
  #       	if exception.errno != errno.EEXIST:
  #          		raise
		idx_list = []
		dim_list = []
		hf_idx = h5py.File(self.gt_path + self.seq_id + '_' + 'idx.h5', 'w')
		hf_lidar = h5py.File(self.gt_path + self.seq_id + '_' + 'lidar.h5', 'w')
		hf_bbox = h5py.File(self.gt_path + self.seq_id + '_' + 'bbox.h5', 'w')

		
		for i in range(self.num_frames):
			frm_data = self.tracklet_data[i]
			img_bgr = self.get_image_data(i)
			all_lidar_pts = self.get_lidar_pts(i)
			# all_lidar_pts = all_lidar_pts[:,:3]

			global_bev = self.gen_bird_view()

			for j in range(len(frm_data)):
				lidar_pts = frm_data[j][:,:8]
				center = frm_data[j][:,8]
				yaw = np.unique(frm_data[j][:,10])[0]
				dim = (frm_data[j][0,[-6,-5,-4]]).tolist()
				dim_list.append(dim)
				rot_mat = np.asarray([[np.cos(yaw), np.sin(yaw), 0.0],[-np.sin(yaw), np.cos(yaw), 0.0],[0.0,0.0, 1.0]])
				pts_in_image = self.project_to_image(lidar_pts)
				bbox = self.get_2d_bbox(pts_in_image)
				
				lidar_dict = self.get_tracklet_pts(all_lidar_pts,rot_mat,center,dim)
				cor_pts = np.array(lidar_dict['lidar_pts'])
				
				if cor_pts.shape[0] < 1000:
					continue				

				bird_view = self.gen_bird_view(cor_pts)
				global_bev = global_bev + bird_view
				
				self.draw_3d_box(pts_in_image,img_bgr)

				lidar_idx = 'lidar_' + self.seq_id + '_' + str(i) + '_' + str(j)
				bbox_idx = 'bbox_' + self.seq_id + '_' + str(i) + '_' + str(j)

				# print lidar_dict['lidar_pts'].shape
				grp_lidar = hf_lidar.create_group(lidar_idx)
				grp_lidar.create_dataset('lidar_pts', data=lidar_dict['lidar_pts'])
				grp_lidar.create_dataset('tf_pts', data=lidar_dict['tf_pts'])
				grp_lidar.create_dataset('center', data=lidar_dict['center'])
				grp_lidar.create_dataset('dims', data=lidar_dict['dims'])
				hf_bbox.create_dataset(bbox_idx, data=bbox)
				idx_list.append((lidar_idx, bbox_idx))

				# self.filtered_lidar_pts.append(cor_pts)
				
				self.draw_rectangle_cv(img_bgr,bbox)
			cv2.imwrite('bird_view_{}_tushar.png'.format(i),global_bev*255)
			cv2.imwrite('res_3d_vis_{}_tushar.png'.format(i),img_bgr)
			self.bev_video.write(np.array(global_bev*255, dtype=np.uint8))
			#self.res_3d_video.write(img_bgr)

		hf_idx.create_dataset('idx', data=idx_list)

		hf_lidar.close()
		hf_bbox.close()		
		hf_idx.close()
		
		#self.bev_video.release()
		#self.res_3d_video.release()
		#print(self.seq_id + '-->', np.array(dim_list).mean(axis=0))
		# with open(self.gt_path + str(self.seq_id) + '.pkl', 'wb') as f:
			# pickle.dump(self.filtered_lidar_pts, f)


if __name__ == '__main__':
	gt_datagen = DataLoader()
	gt_datagen.get_all_annts()
