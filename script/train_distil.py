import sys
sys.path.append(".")
from script.util import gen_ResNetVlad, AverageMeter
from dataset.gen_dataset import ImageList
from dataset.gen_distil_dataset import DistilImageList

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
from torch.autograd import Variable
from torchvision import transforms
import time




def train_distil(batch_size, len128):
    use_cuda = True
    data_len = 4096
    if len128:
        data_len = 128
    student = gen_ResNetVlad(len128,use_cuda)
    student.train()
    image_size = (384, 384)
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
        ])

    vector_size = (240, 320)
    vector_dir = '../netvlad_distil/image_2_vector' + '_' + str(vector_size[0]) + '_' + str(vector_size[1])
    vector_dir = vector_dir.strip()

    # train_loader = torch.utils.data.DataLoader(
        # DistilImageList(root='/datasets/google_landmark/index/google_landmarks/', 
        #     fileList='/datasets/google_landmark/index/netvlad/google_lanmarks_flist.txt.32', 
        #     pklroot='/datasets/google_landmark/index/netvlad/teaching_result/',transform=train_transform), 
        # batch_size=batch_size, shuffle=False,
        # num_workers=4, pin_memory=True, drop_last=False)
    train_loader = torch.utils.data.DataLoader(
        DistilImageList(root='../image_2/', 
            fileList='../image_2_flist.txt', 
            pklroot=vector_dir,transform=train_transform), 
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False)

    criterion = nn.MSELoss().cuda()
    #criterion = StableMSELoss().cuda()
    #criterion = nn.CosineEmbeddingLoss().cuda()
    optimizer = torch.optim.SGD(student.parameters(), 0.1, momentum=0.8, weight_decay= 1e-6)
    for name,p in student.named_parameters():
        print( name, p.shape)
    losses = AverageMeter()

    log_file_name  = 'output_dir/log/train_test.txt'
    print('len128 {}'.format(len128))

    f = open(log_file_name, 'w')
    f.write('start time {}\n'.format(time.time()))
    f.close()
    for epoch in range(0,500):
        print('start epoch {}'.format(epoch))
        if (epoch+1) % 66 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
        for batch_idx, (data,pkldata) in enumerate(train_loader):
            iteration = batch_idx
            data = torch.round(data*255)
            data = data.cuda()
            output = student(data)
            pkldata = pkldata[:,:data_len]
            pkldata = F.normalize(pkldata,p=2,dim=1)
            labels = pkldata.cuda()


            loss = criterion(output, labels )*data_len

            losses.update(loss.item(), data.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            
            if (batch_idx % 500 == 0) or (batch_idx<500 and (batch_idx%5==0)):
                print('{batch_offset:d}: Loss {loss.val:.4f}({loss.avg:.4f})\t'.format(batch_offset=batch_idx, loss=losses))


                f = open(log_file_name, 'a')
                f.write('{batch_offset:d}: Loss {loss.val:.4f}({loss.avg:.4f})\n'.format(batch_offset=batch_idx, loss=losses))
                f.close()
        
        print('{batch_offset:d}: Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(batch_offset=epoch, loss=losses))
        f = open(log_file_name, 'a')
        f.write('{batch_offset:d}: Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(batch_offset=epoch, loss=losses))
        f.close()

        if (epoch+1) % 10 == 0: 
            if use_cuda and (torch.cuda.device_count()) > 1:
                print('Multi_gpu')
                torch.save(student.module.state_dict(), './output_dir/snapshot/student_net_params_epoch_{epoch}.pkl'.format(epoch=epoch))
            else:
                print('Single_gpu')
                torch.save(student.state_dict(), './output_dir/snapshot/student_net_params_epoch_{epoch}.pkl'.format(epoch=epoch))


def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser(description="train_distil")
    parser.add_argument('-len128', action='store_true', default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    train_distil(16,args.len128)
    
