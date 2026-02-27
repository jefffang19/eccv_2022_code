import os
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import sklearn.metrics
from dataset import get_dataset_nih

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine, context):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        
        shape = context.get_binding_shape(i)
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(i):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    
    # Synchronize the stream
    stream.synchronize()
    
    return [out.host for out in outputs]

def main():
    nih_nodule_dataset = 'dataset/{}/nodule/label.csv'
    nih_normal_dataset = 'dataset/{}/normal/label.csv'
    
    engine_file = "model.engine"
    if not os.path.exists(engine_file):
        print(f"Error: {engine_file} not found. Please build it with trtexec first.")
        return

    print(f"Loading TensorRT engine from {engine_file}...")
    with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    print("Engine loaded successfully.")

    print("Loading dataset...")
    _, _, test_loader = get_dataset_nih(nih_nodule_dataset, nih_normal_dataset, batch=4)
    
    context = engine.create_execution_context()

    ys = []
    prds = []
    
    print("Starting inference...")
    
    # Identify index of "whole_predict" output in the model bindings
    output_names = [engine.get_binding_name(i) for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
    print(f"Engine output names: {output_names}")
    if 'whole_predict' in output_names:
        whole_predict_idx = output_names.index('whole_predict')
    else:
        whole_predict_idx = -1  # Fallback to last output
        
    for i, (x, y, _, _) in enumerate(test_loader):
        batch_size = x.shape[0]
        
        # Set dynamic shape for inputs
        context.set_binding_shape(0, (batch_size, 17, 3, 300, 300))
        context.set_binding_shape(1, (batch_size,))
        
        # Allocate buffers based on the set dynamic shape
        inputs, outputs, bindings, stream = allocate_buffers(engine, context)
        
        # Fill inputs: First input is image (x), Second is dummy class
        np.copyto(inputs[0].host, x.numpy().ravel())
        cls_dummy = np.zeros((batch_size,), dtype=np.int64) 
        np.copyto(inputs[1].host, cls_dummy.ravel())
        
        # Run TRT inference
        trt_outputs = do_inference(context, bindings, inputs, outputs, stream)
        
        # Extract whole image predictions
        whole_predict = trt_outputs[whole_predict_idx].reshape(batch_size, 2)
        
        # Apply softmax
        whole_predict = np.exp(whole_predict) / np.sum(np.exp(whole_predict), axis=1, keepdims=True)
        
        # Save results for evaluation
        ys.extend(y.numpy().astype(np.int32))
        prds.extend(whole_predict[:, 1].astype(np.float32))
        
        # Free device memory 
        for inp in inputs:
            inp.device.free()
        for out in outputs:
            out.device.free()

    ys = np.array(ys)
    prds = np.array(prds)
    
    _auc = sklearn.metrics.roc_auc_score(ys, prds, multi_class='ovr')
    _pr = sklearn.metrics.average_precision_score(ys, prds)
    
    print("AUC score on NIH test set via TensorRT : {}".format(_auc))
    print("PR score on NIH test set via TensorRT : {}".format(_pr))

if __name__ == "__main__":
    main()
