import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
from ipdb import set_trace as brk

class VFELayer(nn.Module):

	def __init__(self,c_in,c_out):
		super(VFELayer, self).__init__()
		self.in_dim = c_in
		self.out_dim = c_out
		
		self.fc = nn.Linear(c_in,c_out//2)
		self.bn = nn.BatchNorm1d(c_out//2)
		self.op = nn.ReLU(inplace=True)
		
		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()



	def forward(self,x,mask):
		x = self.fc(x)
		x = x.permute(0,2,1)
		x = self.bn(x)
		x = x.permute(0,2,1)
		x = self.op(x)
		global_feature,___ = torch.max(x,1)
		# print (global_feature.size())
		global_feature = global_feature.unsqueeze(1).expand(x.size()[0],x.size()[1],self.out_dim//2)
		x = torch.cat((x,global_feature),2)
		mask = mask.unsqueeze(2).expand(x.size()[0],x.size()[1],self.out_dim).type(torch.FloatTensor)
		
		# print (x.size())
		return x*mask

class VFE(nn.Module):

	def __init__(self,num_layers=2):
		super(VFE, self).__init__()
		layers=[]
		self.h = 20
		self.w = 10
		'''
		Static Initialization of Dims
		'''

		if num_layers > 2:
			raise Exception('Not Supported')

		dims = [(7,32),(32,128)]
		# for i in range(num_layers):
		# 	layers.append(VFELayer(dims[i][0],dims[i][1]))
			
		#self.layers = nn.Sequential(*layers)
		self.layer_1 = VFELayer(dims[0][0],dims[0][1])
		self.layer_2 = VFELayer(dims[1][0],dims[1][1])
		self.final_fc = nn.Linear(dims[num_layers-1][1],128)
		self.final_bn = nn.BatchNorm1d(128)
		self.final_op = nn.ReLU(inplace=True)
		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()



	def forward(self,x,mask,indices,output):
		x = self.layer_1(x,mask)
		x = self.layer_2(x,mask)
		x = self.final_fc(x)
		x = x.permute(0,2,1)
		x = self.final_bn(x)
		x = x.permute(0,2,1)
		x = self.final_op(x)

		voxel_feature,___ = torch.max(x,1)
		'''
		Transform this tensor n X 128 to [batch_size *  h * w * 128]
		Procedure
		1) use indices from datagen and do scatter_ over torch.zeros(batch_size*h*w, 128)
		2) Perform torch.view(batch_size,h,w,128)
		'''

		voxel_map =output.scatter_(0,indices,voxel_feature).view(-1,self.h,self.w,128)

		return voxel_map


class DenseLidarNet(nn.Module):

	def __init__(self,max_pts_in_voxel=10):
		super(DenseLidarNet, self).__init__()
		self.max_pts_in_voxel= max_pts_in_voxel

		self.vfe_output = VFE()
		self.final_body = self._make_body(4)

		self.x_op = self._make_head()
		self.y_op = self._make_head()
		self.z_op = self._make_head()

		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


	def _make_body(self,num_layers):
		layers=[]
		layers.append(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
		layers.append(nn.ReLU(True))
		for _ in range(num_layers-1):
			layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
		return nn.Sequential(*layers)

	def _make_head(self):
		layers=[]
		layers.append(nn.Conv2d(256, self.max_pts_in_voxel, kernel_size=1, stride=1, padding=0))

		return nn.Sequential(*layers)

	def forward(self,x,mask,indices,output):

		vfe_op = self.vfe_output(x,mask,indices,output)
		vfe_op = vfe_op.transpose(1,3)
		body_op = self.final_body(vfe_op)
		x_op = self.x_op(body_op)
		y_op = self.y_op(body_op)
		z_op = self.z_op(body_op)
		return torch.cat((x_op.view(-1,20*10*self.max_pts_in_voxel, 1),y_op.view(-1,20*10*self.max_pts_in_voxel,1),z_op.view(-1,20*10*self.max_pts_in_voxel,1)),2)


# net = VFELayer(2,2)
# net  = VFE(2)
# mask = torch.randn(10,35) > 0.5
# print (mask.size())
# out = net.forward(Variable(torch.randn(10,35,7)),Variable(mask))
# print (out.size())
