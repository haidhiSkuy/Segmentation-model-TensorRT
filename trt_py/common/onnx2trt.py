import tensorrt as trt 
from utils.logger import MyLogger


def convert_onnx_to_enginer(onnx_path : str, output_name : str):
    logger = MyLogger() 
    builder = trt.Builder(logger)
    network = builder.create_network() 
    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file(onnx_path)

    config = builder.create_builder_config() 
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  

    serialized_engine = builder.build_serialized_network(network, config) 

    with open(output_name, "wb") as f:
        f.write(serialized_engine)