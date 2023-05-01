import cv2
import numpy as np
import sys
from openvino.inference_engine import IENetwork, IECore
# import tensorflow as tf

import time
import datetime

layer = '/Softmax/FlattenONNX_/Reshape'

np.set_printoptions(precision=2, suppress=True)

ie = IECore()
net = ie.read_network(model="saved_model.xml", weights="saved_model.bin")
#net = ie.read_network(model="Lenet5.xml", weights="Lenet5.bin")
#net.add_outputs(layer)
net.add_outputs('StatefulPartitionedCall/sequential/dense_2/BiasAdd/Add')
#net.add_outputs('/layers/layers.11/Gemm')
input_data = np.load("mnist_sample7.npy")
#input_data = np.random.rand(28,28)
input_data = input_data.reshape(1, 1, 28, 28).astype('float16')
input_data /= 255
#print(input_data)
#input_data = input_data.transpose(0, 3, 1, 2)

#print(input_data[0])

input_blob = next(iter(net.input_info))
exec_net = ie.load_network(network=net, device_name="MYRIAD")

print(datetime.datetime.now())
time_sta = time.perf_counter() # Timer start

for i in range(10000):
    out = exec_net.infer(inputs={input_blob: input_data})

time_end = time.perf_counter() # Timer stop
tim = time_end- time_sta
print(datetime.datetime.now())
#request_status = exec_net.requests[0].wait()
"""
res = exec_net.requests[0].output_blobs[""]
print(res.buffer[0])
"""

res = exec_net.requests[0].output_blobs
for i in res:
    out = exec_net.requests[0].output_blobs[i].buffer[0]
    print(i ,out)




#print(out[layer])
#np.savetxt("my_np_" + "Dence21" + "_ir.txt", out[layer][0], fmt = '%3.g')
#np.savetxt("my_np_" + "Conv2D_17" + "_ir.txt", out[layer][0][0], fmt = '%3.g')
print(tim)