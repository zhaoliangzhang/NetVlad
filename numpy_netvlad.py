import numpy as np
import math

def normalize(x):
    for i in range(576):
        max = 0
        for j in range(512):
            max = max + x[j*576+i]*x[j*576+i]
        max = max ** 0.5
        if max < 1e-12:max = 1e-12
        for j in range(512):
            x[j*576+i] = x[j*576+i]/max

    return x

def conv(x, w):
    y = np.zeros((64*24*24,))
    for i in range(64):
        for j in range(576):
            for k in range(512):
                y[i*576+j] = y[i*576+j] + x[k*576+j]*w[512*i+k]
                
    return y

def softmax(x):
    for i in range(576):
        sum = 0
        for j in range(64):
            x[j*576+i] = x[j*576+i] - 360
            sum = sum + math.exp(x[j*576+i])
        for j in range(64):
            x[j*576+i] = math.exp(x[j*576+i])/sum
    return x

def vlad_core(x, s, c):
    y = np.zeros((64*512*576,))
    for i in range(64):
        for j in range(512):
            for k in range(576):
                y[i*576*512+j*576+k] = x[j*576+k] + c[i*512+j]
                y[i*576*512+j*576+k] *= s[576*i+k]

    v = np.zeros((64*512,))
    for i in range(64):
        for j in range(512):
            for k in range(576):
                v[i*512+j] += y[i*512*576+j*576+k]

    return v

def normalize2(x):
    y = np.zeros((32768,))
    for i in range(64):
        sum = 0
        for j in range(512):
            sum = sum + x[j+i*512]*x[j+i*512]
        sum = sum ** 0.5
        if sum < 1e-12:sum = 1e-12
        for j in range(512):
            y[j*64+i] = x[j+i*512]/sum
            #y[j+i*512] = x[j+i*512]/sum

    sum = 0
    for i in range(32768):
        sum = sum + y[i]*y[i]
    sum = sum ** 0.5
    for i in range(32768):
        y[i] = y[i]/sum
    
    return y

def fc(x, w, b):
    y = np.zeros((4096,))
    for i in range(4096):
        for j in range(32768):
            y[i] = y[i] + x[j]*w[i*32768+j]
        y[i] = y[i] + b[i]

    return y

def normalize3(x):
    sum = 0
    for i in range(4096):
        sum = sum + x[i]*x[i]
    sum = sum ** 0.5
    for i in range(4096):
        x[i] = x[i]/sum
    
    return x

input_path = '/home/zzl/Documents/netvlad_apr/output_dir/float_output/'
weight_path = '/home/zzl/Documents/netvlad_apr/models/weights/'
input = np.loadtxt(input_path + 'result.txt', dtype = float)

input_normalized = normalize(input)

conv_weight = np.loadtxt(weight_path + 'conv_weight.txt', dtype = float)
conv_weight = conv_weight.reshape((conv_weight.size,))
conv_result = conv(input_normalized, conv_weight)

after_softmax = softmax(conv_result)

centroids = np.loadtxt(weight_path + 'vlad_centroids.txt', dtype = float).reshape((64*512,))
vlad = vlad_core(input_normalized, after_softmax, centroids)
vlad = normalize2(vlad)

WPVA_weight = np.loadtxt(weight_path + 'WPCA_weight.txt', dtype = float).reshape((4096*32768,))
WPVA_bias = np.loadtxt(weight_path + 'WPCA_bias.txt', dtype = float)

after_fc = fc(vlad, WPVA_weight, WPVA_bias)

output = normalize3(after_fc)
np.savetxt('output_dir/float_output/vlad_output.txt', output.flatten(), fmt='%f')
