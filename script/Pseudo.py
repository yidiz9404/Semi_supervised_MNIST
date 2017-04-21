from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pandas as pd

from sub import subMNIST

#input data with augmentation
trainset_import = pickle.load(open("trainset_import_add_et.p", "rb"))
validset_import = pickle.load(open("validation.p", "rb"))
trainset_unl = pickle.load(open("trainset_unl_add_et.p", "rb"))

train_loader = torch.utils.data.DataLoader(trainset_import, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(validset_import, batch_size=64, shuffle=True)
train_unl_loader = torch.utils.data.DataLoader(trainset_unl, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20,40,kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(360, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50,10)
        self.bnf1 = nn.BatchNorm2d(10)
        self.bnf2 = nn.BatchNorm2d(20)
        self.bnf3 = nn.BatchNorm2d(40)
        self.noise_std = 0

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bnf1(self.conv1(x+ Variable(self.noise_std * torch.randn(x.size())))), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.bnf2(self.conv2(x+ Variable(self.noise_std * torch.randn(x.size()))))), 2))
        #x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = self.conv3_drop(self.bnf3(self.conv3(x+ Variable(self.noise_std * torch.randn(x.size())))))
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        
        
        #x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        
        return F.log_softmax(x)

model = Net()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

"""
semi-supervised learning
"""
time = 0

def alpha(t):
    alpha_f = 3.
    T1 = 100
    T2 = 600
    if t < T1:
        return 0
    elif t < T2:
        return alpha_f * (t-T1) / (T2-T1)
    else:
        return alpha_f

def output2p_label(output):
    """
    get the pseudo label from the model output 
    
    @param output: a Variable
    @return indices: a Variable, squeezed
    """
    y_i, indices = torch.max(output, 1)
    indices = indices.squeeze()
    indices = Variable(indices.data)
    return indices
    
def train_semi(epoch):
    global time
    model.train()
    
    n_labeled = len(train_loader.dataset.train_data)
    n_unlabeled = len(train_unl_loader.dataset.train_data)
    labeled_batch_size = train_loader.batch_size
    unlabeled_batch_size = train_unl_loader.batch_size

    # because #label < #unlabel, use round-robin for labeled data
    labeled_batches = int(np.ceil(1. * n_labeled / labeled_batch_size))
    batches = int(np.ceil(1. * n_unlabeled / unlabeled_batch_size))

    # locally allocate memory to store labeled data list, reduce overhead
    label_list = list(enumerate(train_loader))

    # unlabeled data has no label, enumeration causes error
    # so manually add 0's for convenience [ignore the label anyway]
    train_unl_loader.dataset.train_labels = torch.zeros(n_unlabeled)
    unlabel_list = list(enumerate(train_unl_loader))
    acc_train_label = []
    correct = 0
    loss_unl = []
    loss_all = []
    # semi-supervised learning both labeled and unlabeled data
    for batch_id in range(batches):
        optimizer.zero_grad()
        time += 1
        # get labeled data to supervised learn
        idx, (data, target) = label_list[batch_id % labeled_batches]
        data, target = Variable(data), Variable(target)
        output = model(data)
        labeled_loss = criterion(output, target)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

        # get unlabeled data to unsupervised learn
        idx, (data, _) = unlabel_list[batch_id]
        data = Variable(data)
        output = model(data)
        pseudo_label = output2p_label(output)
        unlabeled_loss = criterion(output, pseudo_label)
  
        
        # sum the losses
        loss = labeled_loss + alpha(time) * unlabeled_loss
        loss.backward()
        optimizer.step()
        

        
        if batch_id % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(train_unl_loader.dataset),
                100. * batch_id / len(train_unl_loader), loss.data[0]))
    acc_train_label = correct/len(train_loader.dataset)
    return acc_train_label

def test(epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    #acc_valid = []
    for data, target in valid_loader:

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    acc_valid = correct / len(valid_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return acc_valid

"""
for test and debugging
"""
for epoch in range(1, 50):
    acc_t = train_semi(epoch)
    acc_val = test(epoch, valid_loader)

testset = pickle.load(open("test.p", "rb" ))
test_loader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=False)
label_predict = np.array([])
model.eval()
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    temp = output.data.max(1)[1].numpy().reshape(-1)
    label_predict = np.concatenate((label_predict, temp))

predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
predict_label.reset_index(inplace=True)
predict_label.rename(columns={'index': 'ID'}, inplace=True)
predict_label.to_csv('sample_submission.csv', index=False)


