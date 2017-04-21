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


class VAE_ConvBn(nn.Module):
    def __init__(self, gamma = 1.):

        super(VAE_ConvBn, self).__init__()

        self.gamma = gamma
        # self.cnl =[]
        cnl = [32, 64, 64, 128]

        self.bn_f1 = nn.BatchNorm2d(cnl[0])
        self.bn_f2 = nn.BatchNorm2d(cnl[1])
        self.bn_f3 = nn.BatchNorm2d(cnl[2])
        self.bn_f4 = nn.BatchNorm2d(cnl[3])

        self.poolf1 = nn.MaxPool2d(2,return_indices=True)
        self.poolf2 = nn.MaxPool2d(2,return_indices=True)

        self.bn_b1 = nn.BatchNorm2d(cnl[3])
        self.bn_b2 = nn.BatchNorm2d(cnl[2])
        self.bn_b3 = nn.BatchNorm2d(cnl[1])
        self.bn_b4 = nn.BatchNorm2d(cnl[0])

        self.unpool1 = nn.MaxUnpool2d(2)
        self.unpool2 = nn.MaxUnpool2d(2)


        self.conv1 = nn.Conv2d(1, cnl[0], kernel_size=5)
        self.conv2 = nn.Conv2d(cnl[0], cnl[1], kernel_size=3)
        self.conv3 = nn.Conv2d(cnl[1], cnl[2], kernel_size=3)
        self.conv4 = nn.Conv2d(cnl[2], cnl[3], kernel_size=3)

        # self.input_size = self.get_input_size()
        self.input_size = torch.randn(1, 28, 28).size()

        self.n_hid = 128

        self.n_latent = 10

        self.fc1 = nn.Linear(self.n_hid, self.n_latent * 2)
        self.fc2 = nn.Linear(self.n_latent, self.n_hid)

        self.dconv1 = nn.ConvTranspose2d(cnl[3], cnl[2], kernel_size=3)
        self.dconv2 = nn.ConvTranspose2d(cnl[2], cnl[1], kernel_size=3)
        self.dconv3 = nn.ConvTranspose2d(cnl[1], cnl[0], kernel_size=3)
        self.dconv4 = nn.ConvTranspose2d(cnl[0], 1, kernel_size=5)

        self.noise_mul = .2


    def encode(self, x):
        if self.training:
            noise = self.noise_mul
        else:
            noise = 0

        x = F.relu(self.bn_f1(self.conv1(x + Variable(noise * torch.randn(x.size())))))
        x, id1 = self.poolf1(x)
        x = F.relu(self.bn_f2(self.conv2(x + Variable(noise * torch.randn(x.size())))))
        x = F.relu(self.bn_f3(self.conv3(x + Variable(noise * torch.randn(x.size())))))
        x, id2 = self.poolf2(x)
        x = F.relu(self.bn_f4(self.conv4(x + Variable(noise * torch.randn(x.size())))))

        x = F.avg_pool2d(x, 2)
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])

        z = self.fc1(x)

        mu = z[:, 0:self.n_latent]
        logvar = z[:, self.n_latent:]

        return mu, logvar , [id1, id2]


    def decode(self, z, idx):

        x = self.fc2(z)
        x = x.view(-1, 128, 1, 1)

        # Unpooling
        x = x.expand(x.size()[0],x.size()[1],2,2)

        x = F.relu(self.bn_b1(x))
        x = F.relu(self.bn_b2(self.dconv1(x)))
        x = self.unpool1(x,idx[1] , output_size=torch.Size([64, 64, 8, 8]))
        x = F.relu(self.bn_b3(self.dconv2(x)))
        x = F.relu(self.bn_b4(self.dconv3(x)))
        x = self.unpool2(x,idx[0] , output_size=torch.Size([64, 32, 24, 24]))
        x = self.dconv4(x)

        return x


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    def forward(self, x):
        mu, logvar, idx = self.encode(x)
        # eps = Variable(torch.randn(logvar.size()))
        # z = mu + torch.exp(logvar / 2) * eps
        z = self.reparametrize(mu, logvar)
        x_hat = self.decode(z, idx)

        return x_hat, mu, logvar


    def sup_loss(self, x, y):

        x_hat, mu, logvar = self.forward(x)
        y_hat = F.log_softmax(mu)

        recon_loss =  recon_loss_function(x_hat, x, mu, logvar)
        loss = F.nll_loss(y_hat, y) + recon_loss * self.gamma

        return loss


    def unsup_loss(self, x):
        x_hat, mu, logvar = self.forward(x)
        loss =  recon_loss_function(x_hat, x, mu, logvar) * self.gamma
        return loss


    def predict(self,x):
        x_hat, mu, logvar = self.forward(x)
        return F.log_softmax(mu)


def recon_loss_function(recon_x, x, mu, logvar):
    # BCE = nn.BCELoss(recon_x, x)
    MSE = (recon_x - x) ** 2
    MSE = MSE.mean()
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = torch.sum(KLD_element).mul_(-0.5)
    return MSE
