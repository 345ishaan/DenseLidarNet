from __future__ import print_function

import os
import argparse
from pdb import set_trace as brk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import time
from logger import Logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from dataloader import DenseLidarGen

# from DenseLidarNet import DenseLidarNet
from chamfer_loss import *
from vfe_layer import *



class Main(object):

	def __init__(self):

		self.batch_size = 2
		self.max_pts_in_voxel = 20
		#normalize  = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)
		self.transform = transforms.Compose([transforms.ToTensor()])
		self.dataset = DenseLidarGen('../../DenseLidarNet_data/all_annt_train.pickle','/home/ishan/images',self.transform)
		self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, collate_fn=self.dataset.collate_fn)
		
		# self.load_model()
		self.h = 20
		self.w = 10
		self.use_cuda = torch.cuda.is_available()        
		self.load_model()
		self.criterion = ChamferLoss()
		self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
		self.run_time = time.ctime().replace(' ', '_')[:-8]
		directory = 'progress/' + self.run_time
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.logger = Logger('directory')

	def plot_stats(self, epoch, data_1, data_2, label_1, label_2, plt):
		plt.plot(range(epoch), data_1, 'r--', label=label_1)
		if data_2 is not None:
			plt.plot(range(epoch), data_2, 'g--', label=label_2)
		plt.legend()

	def load_model(self):
		self.net  = DenseLidarNet()
		self.net.load_state_dict(torch.load('./scripts/net.pth'))
		# assert torch.cuda.is_available(), 'Error: CUDA not found!'
		
		# self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
		if self.use_cuda:
			self.net.cuda()
        
	def adjust_learning_rate(self, optimizer, epoch, base_lr):
		"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
		lr = base_lr * (0.1 ** (epoch // 1000))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
            
	def train(self, epoch):
		train_loss = []
		for batch_idx,(voxel_features,voxel_mask,voxel_indices, chamfer_gt) in enumerate(self.dataloader):
			voxel_features = Variable(voxel_features)
			voxel_mask = Variable(voxel_mask.squeeze()).cuda() 
			voxel_indices = Variable(voxel_indices.unsqueeze(1).expand(voxel_indices.size()[0],128))
			vfe_output = Variable(torch.zeros(self.batch_size*self.h*self.w,128))
            
			if self.use_cuda:
				voxel_features = voxel_features.cuda()
				voxel_mask = voxel_mask.cuda()
				voxel_indices = voxel_indices.cuda()
				vfe_output = vfe_output.cuda()
			#brk()
			xyz_output= self.net.forward(voxel_features,voxel_mask,voxel_indices,vfe_output)
			#print(xyz_output)
			loss = self.criterion(xyz_output, Variable(chamfer_gt).cuda() if self.use_cuda else Variable(chamfer_gt))
			train_loss += [loss.data[0]]
			print('train_loss EPOCH [%d] ITER [%d]  %.3f' % (epoch ,batch_idx,loss.data[0]))
			self.optimizer.zero_grad()

			loss.backward()
			self.optimizer.step()

			torch.save(self.net.state_dict(), '../../model_state.pth')
			torch.save(self.optimizer.state_dict(), '../../opt_state.pth')
            
			#for param in self.net.parameters():
			#	print(param.data)			
		return train_loss


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Dense LiDarNet Training')
	parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
	parser.add_argument('--resume', '-r', default=False, type=bool, help='resume from checkpoint')
	parser.add_argument('--epochs', default=10000, type=int, metavar='N', help='number of total epochs to run')
	parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
	parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
	args = parser.parse_args()
    
	print ("****************************************************************************************")
	print ("Using Learning Rate     ==============> {}".format(args.lr))
	print ("Loading from checkpoint ==============> {}".format(args.resume))
	print ("GPU processing available : ", torch.cuda.is_available())
	print ("Number of GPU units available :", torch.cuda.device_count())
	print ("****************************************************************************************")

	net = Main()
	train_loss = []
	for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
		net.adjust_learning_rate(net.optimizer, epoch, args.lr)
        
        # uncomment following two series of code block to verify if backpropgation is happening properly
		# old_params = []
		# for i in range(len(list(net.net.parameters()))):
		# 	old_params.append(list(net.net.parameters())[i])
            
		train_stats = net.train(epoch)
		train_loss += [np.mean(train_stats)]

		# for i in range(len(list(net.net.parameters()))):
		# 	print("weight update for parameter : ", i, not torch.equal(old_params[i].data, list(net.net.parameters())[i].data))

		plt.figure(figsize=(12,12))
		net.plot_stats(epoch+1, train_loss, None, 'train_loss', 'val_loss', plt)
		plt.savefig('progress/' + net.run_time + '/stats.jpg')
		plt.clf()

