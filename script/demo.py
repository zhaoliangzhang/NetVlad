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
from models.netdef_resnetvlad_4096 import ResNetVlad as ResNetVlad4096
from models.netdef_originnetvlad import OriginNetVlad
import torch._utils
from PIL import Image
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

caffemodel_dir = '/home/zzl/Documents/netvlad_kitti/deephi_netvlad/transformed/'
imagepath = '/home/zzl/Documents/netvlad_kitti/test_data/'
input_size = (1, 3, 384, 384)

model_def = os.path.join(caffemodel_dir, 'model.prototxt')
model_weights = os.path.join(caffemodel_dir, 'model.caffemodel')
input_name = 'ConvNdBackward1'
output_name = 'ConvNdBackward29'
# output_name = 'MaxPool2dBackward4'
# pytorch net
# model = torchvision.models.inception_v3(pretrained=True, transform_input=False)
model = OriginNetVlad(back_front_all="front")
print('1')

#state_dict = torch.load('/home/yujc/netvlad/netvlad_kitti/models/weights/resnetvlad128.pkl.epoch9')
#del state_dict['encoder.bn1.num_batches_tracked']
state_dict = torch.load('/home/zzl/Documents/netvlad_kitti/models/weights/originNetVlad.pkl')
model_dict = model.state_dict()

#pretrained_dict =  {k: v for k, v in state_dict.items() if k in model_dict}
#print("1.5")
#print(pretrained_dict)
#model_dict.update(pretrained_dict)

model.load_state_dict(state_dict)
model.eval()

# random input
# image = np.random.randint(0, 255, input_size)
# image = 66*np.ones(input_size)
image = np.array(Image.open(imagepath + os.listdir(imagepath)[0]).resize((384,384)))
image = image.transpose((2,0,1))[np.newaxis,:]
print(image.shape)
input_data = image.astype(np.float32)

# pytorch forward
input_var = Variable(torch.from_numpy(input_data))

if not test_mod:
    # generate caffe model
    output_var = model(input_var)
    print(output_var.size())
    #plot_graph(output_var, os.path.join(caffemodel_dir, 'pytorch_graph.dot'))
    pytorch2caffe(input_var, output_var, model_def, model_weights)
    exit(0)

# test caffemodel
# caffe.set_device(0)
caffe.set_mode_cpu()
net = caffe.Net(model_def, model_weights, caffe.TEST)

net.blobs['data'].data[...] = input_data
net.forward(start=input_name)
caffe_output = net.blobs[output_name].data

model = model.cpu()
input_var = input_var.cpu()
output_var = model(input_var)
pytorch_output = output_var.data.cpu().numpy()

print(input_size, pytorch_output.shape, caffe_output.shape)
print('pytorch: min: {}, max: {}, mean: {}'.format(pytorch_output.min(), pytorch_output.max(), pytorch_output.mean()))
print('  caffe: min: {}, max: {}, mean: {}'.format(caffe_output.min(), caffe_output.max(), caffe_output.mean()))

#save data
shape = caffe_output.shape
np.savetxt('../output_dir/float_output/caffe_output.txt', caffe_output.flatten(), fmt='%f')
np.savetxt('../output_dir/float_output/pytorch_output.txt', pytorch_output.flatten(), fmt='%f')

diff = np.abs(pytorch_output - caffe_output)
print('   diff: min: {}, max: {}, mean: {}, median: {}'.format(diff.min(), diff.max(), diff.mean(), np.median(diff)))
