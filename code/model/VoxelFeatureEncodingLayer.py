import torch
import torchvision.models as models
import torch.nn.functional as F

class VFELayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        self.fc = nn.Linear(input_dim, output_dim//2)
        self.bn = nn.BatchNorm1d(output_dim//2)
        self.op = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.fc(x)
        output = self.bn(output)
        output = self.op(output)
        element_wise_maxpool = torch.max(output,0)[0]
        sz = output.size()
        output = \
            torch.cat(
                (element_wise_maxpool.expand(sz[0],sz[1]), output), 
                1
            )
        return output