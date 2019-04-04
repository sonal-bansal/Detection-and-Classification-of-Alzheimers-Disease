from torchvision import transforms, datasets
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import dataset, dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from Recog_modelv1 import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
print(model)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


data_transform = transforms.Compose([
        transforms.ToTensor()
])


model.load_state_dict(torch.load('\dev\Alzheimers\Saved_Model\RecogModelv4.pt'))
test_data = torchvision.datasets.ImageFolder("\dev\Alzheimers\seperated-data\resized_val", transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle = False, num_workers=4)

correct = 0
total = 0
pred=[]
label=[]
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs[0][0].reshape(1,1,inputs.shape[2],inputs.shape[3]))
        loss = criterion(outputs, labels)
        print(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        pred.append(predicted)
        label.append(labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('Accuracy of the test images: %d %%' % (100 * correct / total))
        print(predicted,labels)

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

# print(total)
# print(pred)
# print(label)

