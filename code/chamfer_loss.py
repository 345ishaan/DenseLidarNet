import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ChamferLoss(nn.Module):

	def __init__(self):
		super(ChamferLoss, self).__init__()
		self.eps = 1e-8

	def forward(self,preds,gts):
		P = self.batch_pairwise_dist(gt, pred)
		mins, _ = torch.min(P, 1)
		loss_1 = torch.sum(mins)
		mins, _ = torch.min(P, 2)
		loss_2 = torch.sum(mins)

		return loss_1 + loss_2


	def batch_pairwise_dist(self,x,y):
		bs, num_points, points_dim = x.size()
		xx = torch.bmm(x, x.transpose(2,1))
		yy = torch.bmm(y, y.transpose(2,1))
		zz = torch.bmm(x, y.transpose(2,1))
		diag_ind = torch.arange(0, num_points).type(torch.LongTensor)
		rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
		ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
		P = (rx.transpose(2,1) + ry - 2*zz)
		return P