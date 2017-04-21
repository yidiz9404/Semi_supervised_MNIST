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
import pickle
import torchvision
import matplotlib.pyplot as plt
import sys
from vae_bn import *
import pandas as pd

train_labeled = pickle.load(open("trainset_import_add_et.p", "rb"))  # using the augmented training dataset
train_unlabeled = pickle.load(open("trainset_unl_add_et.p", "rb"))
# train_unlabeled.train_labels = torch.ones([47000])
# train_unlabeled.k = 47000
val_data = pickle.load(open("validation.p", "rb"))
test_data = pickle.load(open("test.p","rb"))


train_loader_unlabeled = torch.utils.data.DataLoader(train_unlabeled, batch_size=64, shuffle=True)
train_loader_labeled = torch.utils.data.DataLoader(train_labeled, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


model = VAE_ConvBn()


opt = optim.Adam(model.parameters(), lr=1e-3)


def train_unsup():
    ttl_loss = 0
    count = 0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader_unlabeled):
        data, target = Variable(data), Variable(target)

        opt.zero_grad()
        # Change to forward + loss
        loss = model.unsup_loss(data)
        loss.backward()
        opt.step()
        ttl_loss += loss
        count += 1

    print("unsup averge loss: ", (ttl_loss / count).data[0])


def train_sup():

    ttl_loss = 0
    count = 0
    correct = 0
    train_acc = []

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader_labeled):
        data, target = Variable(data), Variable(target)
        opt.zero_grad()
        loss = model.sup_loss(data, target)
        loss.backward()
        opt.step()
        #avg_forward_loss += class_loss
        ttl_loss += loss

        y_hat = model.predict(data)
        pred = y_hat.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        count += 1

    # np.save('train_acc.npy', train_acc)
    print("sup averge loss: ", (ttl_loss / count).data[0])
    return 100. * correct / len(train_loader_labeled.dataset)


def test(record=False):
    test_loss = 0
    correct = 0
    right = []
    val_acc = []

    model.eval()

    for data, target in val_loader:

        data, target = Variable(data, volatile=True), Variable(target)
        y_hat = model.predict(data)
        loss = model.sup_loss(data,target)
        test_loss += loss.data[0]
        pred = y_hat.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(val_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    # np.save('val_acc.npy', val_acc)
    return 100. * correct / len(val_loader.dataset)


def predict_test_data():

    label_predict = np.array([])
    model.eval()

    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model.predict(data)
        temp = output.data.max(1)[1].numpy().reshape(-1)
        label_predict = np.concatenate((label_predict, temp))

    predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
    predict_label.reset_index(inplace=True)
    predict_label.rename(columns={'index': 'ID'}, inplace=True)
    predict_label.to_csv('submission_overnight.csv', index=False)


train_acc = []
val_acc = []

for i in range(150):

    train_acc.append(train_sup())
    # train_unsup()
    val_acc.append(test())

np.save('train_acc.npy', train_acc)
np.save('val_acc.npy', val_acc)
print("AVERAGE VALIDATION ACCURACY:")
print(np.array(val_acc).mean() )

predict_test_data()
