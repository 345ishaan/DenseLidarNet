import torch

def batch_pairwise_dist(x,y):
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    diag_ind = torch.arange(0, num_points).type(torch.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2,1) + ry - 2*zz)
    return P


def chamfer_distance(gt, pred):
    
    P = batch_pairwise_dist(gt, pred)
    mins, _ = torch.min(P, 1)
    loss_1 = torch.sum(mins)
    mins, _ = torch.min(P, 2)
    loss_2 = torch.sum(mins)

    return loss_1 + loss_2