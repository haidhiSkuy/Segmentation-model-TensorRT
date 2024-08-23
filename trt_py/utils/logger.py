import tensorrt as trt 

logger = trt.Logger(trt.Logger.WARNING) 

class MyLogger(trt.ILogger):
    def __init__(self):
       trt.ILogger.__init__(self)

    def log(self, severity, msg):
        pass # Your custom logging implementation here