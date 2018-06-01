import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class iOULoss(nn.Module):
    
    def __init__(self):
        super(iOULoss, self).__init__()
        self.eps = 1e-8

    def forward(self,preds,gts):
        
        corner_low, corner_high, area_preds, area_gts = self.compute_intersection(preds,gts)
        corner_diff = corner_high - corner_low
        intersection = torch.prod(corner_diff,1)
        union = area_gts + area_preds - intersection
        iou = intersection/union
        
        return iou
    
    def compute_intersection(self, preds, gts):
        
        corner_preds_low, _ = torch.min(preds, 1)
        corner_preds_high, _ = torch.max(preds, 1)
        
        corner_gts_low, _ = torch.min(gts, 1)
        corner_gts_high, _ = torch.max(gts, 1)

        corner_low, _ = torch.max(torch.stack([corner_preds_low, corner_gts_low]), 0)
        corner_high, _ = torch.min(torch.stack([corner_preds_high, corner_gts_high]), 0)
        
        corner_preds_diff = corner_preds_high - corner_preds_low
        corner_gts_diff = corner_gts_high - corner_gts_low
        
        area_preds = torch.prod(corner_preds_diff, 1)
        area_gts = torch.prod(corner_gts_diff, 1)
        
        return corner_low, corner_high, area_preds, area_gts
        