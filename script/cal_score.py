import sys
sys.path.append(".")
import argparse
from script.util import gen_ResNetVlad, gen_OriginVlad, get_fix_feature, gen_backend, gen_frontend, gen_origin_frontend, gen_origin_backend
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
import caffe
import unittest
from sklearn import preprocessing
from scipy.spatial.distance import pdist

from torch.autograd import Variable
from dataset.gen_dataset import ImageList
from torchvision import transforms
import torch.nn.functional as F


def get_labels(batch_size,origin,len128,output_dir,fixtest=False,fixorigin=False):
    descs = []
    #teaching = gen_originNetVlad()
    if fixtest:
        if origin:
            frontend = gen_origin_frontend(True)
            frontend.eval()
            backend = gen_origin_backend(True)
            backend.eval()
        else:
            frontend = gen_frontend(len128, True)
            frontend.eval()
            backend = gen_backend(len128, True)
            backend.eval()
    elif origin:
        teaching = gen_OriginVlad(True)
        teaching.eval()
    else:
        teaching = gen_ResNetVlad(len128, True)
        teaching.eval()
    
    image_size = (384, 384)
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
        ])
    train_loader = torch.utils.data.DataLoader(
        ImageList(root='/home/zzl/Documents/netvlad_kitti/test_data/', fileList='./image_2_flist.txt',transform=train_transform),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False)

    print('hello')
    for batch_idx, (data,imgPaths) in enumerate(train_loader):
        # print (batch_idx)
        # print ((data,imgPaths))
        iteration = batch_idx
        print(iteration)
        print(data.shape)

        if fixtest:
            assert(len(imgPaths) == 1)
            features = get_fix_feature(imgPaths[0],len128)
            data = torch.from_numpy(features)
            frontresult = data.unsqueeze(0)
            # data = torch.round(data*255)
            # data = data.cuda()
            # frontresult = frontend(data)
            output = backend(frontresult)
            # print(imgPaths[0])
            # exit(0)
            for i in range(len(imgPaths)):
                outputfile = output_dir + ( "/batch_res128/" if len128 else "/batch_res4096/") + imgPaths[i] 
                outputfile = (outputfile + "fix.plk") if fixtest else (outputfile+".plk")
                torch.save(frontresult[i],outputfile)

        else:
            data = torch.round(data*255)
            data = data.cuda()
            # print(data.shape)
            output = teaching(data)
        
        
       

        descs = descs + output.tolist()
    return descs

def get_matrix(feats, nounce, use_dim, output_dir ):
    
    output = open(output_dir + '/feats_00_{}_{}.pkl'.format(nounce,use_dim), 'wb')
    pickle.dump(feats, output)
    output.close()
    print("length is {}".format(use_dim))
    use_feats = np.array(feats)[:, :use_dim]
    use_feats = torch.from_numpy(use_feats)
    use_feats = F.normalize(use_feats,p=2,dim=1)
    use_feats = use_feats.numpy()
    #########################################
    score = np.dot(use_feats,use_feats.T)
    
    output = open(output_dir + '/feats_score_00_{}_{}.pkl'.format(nounce,use_dim), 'wb')
    pickle.dump(score, output)
    output.close()

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser(description="get_matrix")
    parser.add_argument('--nounce',
                        default='float_4096')
    parser.add_argument('--output_dir',
                        default='./output_dir/val_reloc/')
    parser.add_argument('-origin', action='store_true', default=False)
    parser.add_argument('-len128', action='store_true', default=False)
    parser.add_argument('-fixtest', action='store_true', default=False)
    parser.add_argument('--use_dim', default=128, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.fixtest:
        descs = get_labels(1,args.origin,args.len128, args.output_dir, args.fixtest)
    else:
        descs = get_labels(16,args.origin,args.len128, args.output_dir, args.fixtest)

    if args.nounce != 'float_4096':
        pass
    elif args.origin:
        args.nounce = 'origin'
    elif args.len128:
        args.nounce = 'float_128'

    if args.fixtest:
        args.nounce += '_fix'
    get_matrix(descs,args.nounce,args.use_dim,args.output_dir)
    
