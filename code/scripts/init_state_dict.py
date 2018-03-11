import sys
import math
import torch
import torch.nn as nn
import torch.nn.init as init
sys.path.insert(0,'../')
from vfe_layer import *

net = DenseLidarNet()
state = net.state_dict()
net.load_state_dict(state)
torch.save(net.state_dict(), 'net.pth')
print('Done!')
