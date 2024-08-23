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