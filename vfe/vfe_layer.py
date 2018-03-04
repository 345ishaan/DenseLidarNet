import torch
import torch.nn as nn
from torch.autograd import Variable



class VFELayer(nn.Module):

	def __init__(self,c_in,c_out):
		super(VFELayer, self).__init__()
		self.in_dim = c_in
		self.out_dim = c_out
		
		self.fc = nn.Linear(c_in,c_out//2)
		self.bn = nn.BatchNorm1d(c_out//2)
		self.op = nn.ReLU(inplace=True)

	def forward(self,x):
		x = self.fc(x)
		x = self.bn(x)
		x = self.op(x)

		global_feature,___ = torch.max(x,0)
		global_feature = global_feature.expand(x.size()[0],self.out_dim//2)
		x = torch.cat((x,global_feature),1)
		return x

class VFE(nn.Module):

	def __init__(self,num_layers=2):
		super(VFE, self).__init__()
		layers=[]
		'''
		Static Initialization of Dims
		'''
		if num_layers > 2:
			raise Exception('Not Supported')
		
		dims = [(7,32),(32,128)]
		for i in range(num_layers):
			layers.append(VFELayer(dims[i][0],dims[i][1]))
		self.layers = nn.Sequential(*layers)
		self.final_fc = nn.Sequential(
						nn.Linear(dims[num_layers-1][1],128),
						nn.BatchNorm1d(128),
						nn.ReLU(inplace=True)
						)


	def forward(self,x):
		x = self.layers(x)
		x = self.final_fc(x)
		voxel_feature,___ = torch.max(x,0)
		return voxel_feature


net  = VFE(3)
out = net.forward(Variable(torch.randn(11,7)))
print (out.size())