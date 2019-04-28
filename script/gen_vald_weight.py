#in_test.prototxt -*- codng: utf-8
import sys
sys.path.append("/home/yujc/caffe-comp/python")
import caffe
import numpy as np
import os

import torch
from torch.autograd import Variable
import torchvision
from models.test_originnetvlad import OriginNetVlad
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

model = OriginNetVlad(back_front_all="back")
weight_path = './models/weights/'

#f = open(weight_path + 'vlad_weight.txt', 'w')

conv_weight = model.state_dict()['net_vlad.conv.weight'].numpy()
N, C = conv_weight.shape[:2]
conv_weight = conv_weight.reshape(N,C)
np.savetxt(weight_path + 'conv_weight.txt', conv_weight, fmt='%f')
vlad_centroids = model.state_dict()['net_vlad.centroids'].numpy()
np.savetxt(weight_path + 'vlad_centroids.txt', vlad_centroids, fmt='%f')
#print(vlad_centroids.shape)
WPCA_weight = model.state_dict()['WPCA.weight'].numpy()
np.savetxt(weight_path + 'WPCA_weight.txt', WPCA_weight, fmt='%f')
#print(WPCA_weight.shape)
WPCA_bias = model.state_dict()['WPCA.bias'].numpy()
np.savetxt(weight_path + 'WPCA_bias.txt', WPCA_bias, fmt='%f')
#print(WPCA_bias.shape)
#z = np.array()
#np.savez(weight_path + 'vlad_weight.txt', conv_weight, vlad_centroids, WPCA_weight, WPCA_bias, fmt='%f')


#f.close()