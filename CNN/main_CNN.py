from CNN_func import *
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

transform = transforms.Compose({
    transforms.ToTensor(),  
})

train_data = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

# test
test_data = MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("length of train：{}".format(train_data_size))
print("length of test：{}".format(test_data_size))

# my test
dataX = sio.loadmat('dataX.mat')
dataX = dataX['dataX']  
dataY = sio.loadmat('dataY.mat')
dataY = dataY['dataY']  

dataX = np.transpose(dataX, (3, 2, 0, 1)) 
dataX = torch.from_numpy(dataX).float() / 1  
dataY = torch.from_numpy(dataY.squeeze()).long()  

custom_dataset = TensorDataset(dataX, dataY)
my_loader = DataLoader(custom_dataset, batch_size=64, shuffle=True)

model = MnistModel()
criterion = nn.CrossEntropyLoss()   

optimizer = torch.optim.SGD(model.parameters(), lr=0.14)#lr:学习率

best_accuracy = 0.0
for epoch in range(5):
    print("Epoch {}/5".format(epoch + 1))

    train(model, train_loader, criterion, optimizer)

    test_accuracy = evaluate_model(model, my_loader)

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), "./model/best_model.pkl")

    print("Test Accuracy: {:.4f}\n".format(test_accuracy))
