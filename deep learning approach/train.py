from torchvision import transforms, datasets
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import dataset, dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
from Recog_modelv1 import Net 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
print(model)

torch.save(model.state_dict(), '/dev/Alzheimers/Saved_Model/RecogModelv3.pt')
# os.popen('gsutil cp -r \dev\Alzheimers\Saved_Model\RecogModelv3.pt gs:\\saved_weight\')
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


data_transform = transforms.Compose([
        transforms.ToTensor()
])



train_data = torchvision.datasets.ImageFolder("/dev/Alzheimers/seperated-data/combined-ad", transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle = True, num_workers=4)


total_step = len(train_loader)
epochs=10
model.train()
for epoch in range(epochs):
    total=0
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):

        inputs=inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs[0][0].reshape(1,1,inputs.shape[2],inputs.shape[3]))
        
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total=total+1
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Total loss : {:.4f}'
            .format(epoch+1,epochs,i+1,total_step, loss.item(), running_loss))
        
        
    torch.save(model.state_dict(), '/dev/Alzheimers/Saved_Model/RecogModelv3.pt')

    print("Total Loss : " + str(running_loss/total))
print('Finished Training')

