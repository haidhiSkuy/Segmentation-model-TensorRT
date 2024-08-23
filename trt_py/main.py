import numpy as np
import tensorrt as trt
from utils.util import *
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

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]["shape"], self.inputs[0]["dtype"]  
    
    def output_spec(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]["shape"], self.outputs[0]["dtype"] 
    
    def infer(self, batch, top=1):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param top: The number of classes to return as top_predicitons, in descending order by their score. By default,
        setting to one will return the same as the maximum score class. Useful for Top-5 accuracy metrics in validation.
        :return: Three items, as numpy arrays for each batch image: The maximum score class, the corresponding maximum
        score, and a list of the top N classes and scores.
        """
        # Prepare the output data
        output_shape, output_dtype = self.output_spec()
        output = np.zeros(output_shape, dtype=output_dtype) # zeros array with size is the model output size (1, 3, 256, 256) 

        # Process I/O and execute the network
        common.memcpy_host_to_device(
            self.inputs[0]["allocation"], np.ascontiguousarray(batch)
        )
        self.context.execute_v2(self.allocations)
        common.memcpy_device_to_host(output, self.outputs[0]["allocation"]) 

        output_mask = sigmoid(output)[0][0]
        output_mask = np.where(output_mask > 0.5, 1, 0) * 255
        return output_mask 
    

if __name__ == "__main__": 
    import cv2
    from image_batcher import ImageBatcher

    input_path = "sample_image/image4.png"

    inference = TensorRTInfer("model/ghost_unet.engine")
    input_shape, input_dtype = inference.input_spec()

    batcher = ImageBatcher(
        input=input_path, 
        shape=input_shape, 
        dtype=input_dtype
    )


    for i, (batch, images) in enumerate(batcher.get_batch()):
        out = inference.infer(batch)
        cv2.imwrite(f"out_{i}.png", out)