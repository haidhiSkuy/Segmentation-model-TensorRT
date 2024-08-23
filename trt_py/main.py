import numpy as np
import tensorrt as trt
from utils.logger import MyLogger 

from common import common
from cuda import cudart

class TensorRTInfer: 
    def __init__(self, engine_path : str) -> None:
        
        self.logger = MyLogger()
        self.runtime = trt.Runtime(self.logger) 

        # Load TRT Engine
        with open(engine_path, "rb") as f:
            serialized_engine = f.read() 
            self.engine = self.runtime.deserialize_cuda_engine(serialized_engine) 
        self.context = self.engine.create_execution_context() 

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []  

        for i in range(self.engine.num_io_tensors): 
            name = self.engine.get_tensor_name(i) 
            is_input = False

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True

            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)

            if is_input:
                self.batch_size = shape[0] 

            size = np.dtype(trt.nptype(dtype)).itemsize 

            for s in shape:
                size *= s 

            allocation = common.cuda_call(cudart.cudaMalloc(size))

            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            } 

            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)