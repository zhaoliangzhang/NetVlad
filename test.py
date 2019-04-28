#in_test.prototxt -*- codng: utf-8
#from pytorch2caffe import plot_graph, pytorch2caffe
import sys
#sys.path.append('/data/build_caffe/caffe_rtpose/python')
sys.path.append("/home/yujc/caffe-comp/python")
import caffe
import numpy as np
import os

import torch
from torch.autograd import Variable
import torchvision
from models.netdef_resnetvlad_128 import ResNetVlad as ResNetVlad128
from script.util import get_fix_feature
from models.netdef_resnetvlad_4096 import ResNetVlad as ResNetVlad4096
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


# test the model or generate model
test_mod = True

imagepath = '/home/zzl/Documents/netvlad_kitti/test_data/'

model = OriginNetVlad(back_front_all="back")

state_dict = torch.load('/home/zzl/Documents/netvlad_apr/models/weights/originNetVlad.pkl')
model_dict = model.state_dict()

model.load_state_dict(state_dict)
model.eval()

image = np.loadtxt('/home/zzl/Documents/netvlad_apr/output_dir/float_output/result.txt')
image = image.reshape(1,512,24,24)
input_data = image.astype(np.float32)

# pytorch forward
input_var = Variable(torch.from_numpy(input_data))


model = model.cpu()
input_var = input_var.cpu()
output_var = model(input_var)
pytorch_output = output_var.data.cpu().numpy()
