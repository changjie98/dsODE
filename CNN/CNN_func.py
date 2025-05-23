import torch
from torch import nn
from torch.nn import Conv2d, Linear, ReLU
from torch.nn import MaxPool2d


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0, bias=False)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=False)
        self.maxpool2 = MaxPool2d(2)
        self.linear1 = Linear(320, 128, bias=False)
        self.linear2 = Linear(128, 10, bias=False)
        #self.linear3 = Linear(64, 10)
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        #x = self.linear3(x)

        return x


class MnistModel_small(nn.Module):
    def __init__(self):
        super(MnistModel_small, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0, bias=False)
        self.maxpool1 = MaxPool2d(4)
        self.linear1 = Linear(360, 128, bias=False)
        self.linear2 = Linear(128, 10, bias=False)
        # self.linear3 = Linear(64, 10)
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        # x = self.linear3(x)

        return x


def train(model,train_loader,criterion,optimizer):
    # index = 0
    for index, data in enumerate(train_loader):
        # for data in train_loader:
       input, target = data   
       y_predict = model(input)
       loss = criterion(y_predict, target)
       optimizer.zero_grad() 
       loss.backward()
       optimizer.step()
       # index += 1
       if index % 100 == 0: 
           torch.save(model.state_dict(), "./model/model.pkl")  
           torch.save(optimizer.state_dict(), "./model/optimizer.pkl")
           

    #return model


#if os.path.exists('./model/model.pkl'):
#   model.load_state_dict(torch.load("./model/model.pkl"))#加载保存模型的参数


def evaluate_model(model, test_loader):
    correct = 0     
    total = 0  
    with torch.no_grad():   
        for data in test_loader:
            input, target = data
            output = model(input)  
            probability, predict = torch.max(input=output.data, dim=1)    
            # loss = criterion(output, target)
            total += target.size(0) 
            correct += (predict == target).sum().item()  
    accuracy = correct / total
    print("accuracy is：%.6f" % accuracy)
    return accuracy 
