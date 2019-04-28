import sys
import os
sys.path.append(".")
from models.netdef_resnetvlad_4096 import ResNetVlad as ResNetVlad4096
from models.netdef_resnetvlad_128 import ResNetVlad as ResNetVlad128
from models.netdef_originnetvlad import OriginNetVlad
from dataset.gen_dataset import ImageList
from dataset.gen_distil_dataset import DistilImageList
import argparse
import torch
import collections
import torch.nn as nn
import scipy.io as sio
import numpy as np
import torch.nn.functional as F


def gen_OriginVlad(use_cuda):
    model = OriginNetVlad()
    state_dict = torch.load('models/weights/originNetVlad.pkl')
    model.load_state_dict(state_dict)
    if use_cuda:
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print('Multi_gpu')
            model = torch.nn.DataParallel(model)

    return model

def gen_ResNetVlad( len128,use_cuda):    
    if len128:
        model = ResNetVlad128()
        state_dict = torch.load('models/weights/resnetvlad128.pkl.epoch9')
    else :
        model = ResNetVlad4096()
        state_dict = torch.load('models/weights/resnetvlad4096.pkl')
    
    model.load_state_dict(state_dict)

    if use_cuda:
        model = model.cuda()
        if torch.cuda.device_count() > 0:
            print('Multi_gpu')
            model = torch.nn.DataParallel(model)
    return model

def gen_backend(len128,use_cuda):
    if len128:
        model = OriginNetVlad(back_front_all="back")
        state_dict = torch.load('models/weights/resnetvlad128.pkl.epoch9')
    else :
        model = ResNetVlad4096( back_front_all="back")
        state_dict = torch.load('models/weights/resnetvlad4096.pkl')
    
    model.load_state_dict(state_dict)
    if use_cuda:
        model = model.cuda()
        if torch.cuda.device_count() > 0:
            print('Multi_gpu')
            model = torch.nn.DataParallel(model)
    return model


def gen_origin_backend(use_cuda):
    
    model = OriginNetVlad(back_front_all="back")
    state_dict = torch.load('models/weights/originNetVlad.pkl')
    model.load_state_dict(state_dict)
    if use_cuda:
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print('Multi_gpu')
            model = torch.nn.DataParallel(model)

    return model

def gen_origin_frontend(use_cuda):
    
    model = OriginNetVlad(back_front_all="front")
    state_dict = torch.load('models/weights/originNetVlad.pkl')
    model.load_state_dict(state_dict)
    if use_cuda:
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            print('Multi_gpu')
            model = torch.nn.DataParallel(model)

    return model

def gen_frontend(len128,use_cuda):
    if len128:
        model = ResNetVlad128(back_front_all="front")
        state_dict = torch.load('models/weights/resnetvlad128.pkl.epoch9')
    else :
        model = ResNetVlad4096( back_front_all="front")
        state_dict = torch.load('models/weights/resnetvlad4096.pkl')
    
    model.load_state_dict(state_dict)
    if use_cuda:
        model = model.cuda()
        if torch.cuda.device_count() > 0:
            print('Multi_gpu')
            model = torch.nn.DataParallel(model)
    return model

def get_fix_feature(filename):
    with open('./deephi_netvlad/test_calib.txt', 'w') as f:
        print("{} 0".format(filename), file=f)
    assert(os.system('cd ./deephi_netvlad && ./testsim.sh test') == 0)
    count = 0
    #featureshape = np.array([512,12,12])
    featureshape = np.array([512,24,24])
    features = np.zeros(featureshape.prod(),dtype=np.float32)
    #Suffix = "_128" if len128 else "_4096"
    Suffix = ''
    with open("./deephi_netvlad/fix_results" + Suffix +"/fix_test.log") as f:                          # open txt file
        for line in f:                                          # check line by line
            datalist = line.split()                             # split one line to a list
            if len(datalist) > 8:                               # jump over void line
                #if datalist[3] == 'net_test.cpp:305]' and datalist[4] == 'Batch' and datalist[6] == 'AddBackward73':
                if datalist[3] == 'net_test.cpp:305]' and datalist[4] == 'Batch' and datalist[6] == 'ConvNdBackward29':
                    features[count] = np.float32( datalist[8] )
                    # print("input %d: %g" %(count,features[count]))
                    count += 1
    assert( count == featureshape.prod() )
    features = features.reshape(featureshape.tolist())
    return features



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
