import argparse
import torch
import collections
import torch.nn as nn
import scipy.io as sio
import numpy as np
import torch.nn.functional as F
import glob
import os
import cv2
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as scid
import scipy.signal as scisig
import time
import unittest
from sklearn import preprocessing
from scipy.spatial.distance import pdist
from torchvision.models import resnet34,resnet18

class ResNetVlad(nn.Module):
    class NetVLAD(nn.Module):
        """NetVLAD layer implementation"""
        def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                    normalize_input=False):
            """
            Args:
                num_clusters : int
                    The number of clusters
                dim : int
                    Dimension of descriptors
                alpha : float
                    Parameter of initialization. Larger value is harder assignment.
                normalize_input : bool
                    If true, descriptor-wise L2 normalization is applied to input.
            """
            super().__init__()
            self.num_clusters = num_clusters
            self.dim = dim
            self.alpha = alpha
            self.normalize_input = normalize_input
            self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), padding=(0, 0), bias=False)
            self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        #     self._init_params()

        # def _init_params(self):
        #     self.conv.weight = nn.Parameter(
        #         (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        #     )
        #     self.conv.bias = nn.Parameter(
        #         - self.alpha * self.centroids.norm(dim=1)
        #     )

        def forward(self, x):
            N, C = x.shape[:2]

            if self.normalize_input:
                x = F.normalize(x, p=2, dim=1)  # across descriptor dim

            # soft-assignment
            soft_assign = self.conv(x).view(N, self.num_clusters, -1)
            soft_assign = F.softmax(soft_assign, dim=1)

            x_flatten = x.view(N, C, -1)

            # calculate residuals to each clusters
            residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) + \
                self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign.unsqueeze(2)
            vlad = residual.sum(dim=-1)

            vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
            vlad = vlad.permute(0,2,1).contiguous()
            vlad = vlad.view(x.size(0), -1)  # flatten
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize


            return vlad

    def __init__(self, back_front_all = "all"):
        super().__init__()
        self.num_clusters = 64
        self.encoder = resnet18(pretrained=True)
        self.base_model = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
        )
        self.dim = len(list(self.base_model.parameters())[-1]) 
        self.net_vlad = self.NetVLAD(num_clusters=self.num_clusters , dim=self.dim, alpha=1.0)
        self.WPCA= nn.Linear(self.num_clusters*self.dim,4096)
        assert(back_front_all in ["all", "back", "front"] )
        self._back_front_all = back_front_all
        # A = torch.load('./models/init_wpca/WPCA.bias.pth')
        # B = torch.load('./models/init_wpca/WPCA.weight.pth')
        # # paramters = self.state_dict()
        # # paramters.items()
        # for name,p  in self.named_parameters():
        #     if ('WPCA' in name):
        #         if('bias' in name):
        #             self.state_dict()[name][...] = A
        #         elif('weight' in name):
        #             self.state_dict()[name][...] = B
        #     if ('net_vlad' in name) or ('WPCA' in name):
        #         p.requires_grad=True
        #     else :
        #         p.requires_grad=True
        

        # self.preL2 = nn.LocalResponseNorm(1024,alpha = 1024*1, beta = 0.5, k = 1e-12)

        # self.netvlad_layer = nn.Sequential(collections.OrderedDict([
        #     ('convsoft',nn.Conv2d(512,64,kernel_size=[1, 1], stride=(1, 1), padding=(0, 0))),
        #     ('softmax1',nn.Softmax2d()),
        #     ('onlybias',self.OnlyBias(512,64))
        # ]))

    def forward(self, x0):
        if (self._back_front_all in [ "all" , "front"]):
            x29 = self.base_model(x0)
        else:
            x29 = x0
        if (self._back_front_all in ["all", "back"]):
            x30 = self.net_vlad(x29)
            x31 = self.WPCA(x30)
            x32 = F.normalize(x31,p=2,dim=1)
        else :
            x32 = x29
        return x32
