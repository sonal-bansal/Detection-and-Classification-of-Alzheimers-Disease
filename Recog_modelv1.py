
from torchvision import transforms, datasets
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import dataset, dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

class Net(nn.Module):
    height,weight=53,53
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.batch1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6,16, 5)
        self.batch2 = nn.BatchNorm2d(16)
#         self.upsample=nn.ConvTranspose2d(1, 1,20)
        self.fc1 = nn.Linear(1016064, 1000)
        self.fc2 = nn.Linear(1000, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 2)
        
    def forward(self, x):

        x = F.relu(self.batch1(self.conv1(x)))
        x = F.relu(self.batch2(self.conv2(x)))
#         a=x.cpu()
#         plt.imshow(a[0][0].detach().numpy())
#         plt.show()
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
