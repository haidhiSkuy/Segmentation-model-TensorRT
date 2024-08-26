#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include <NvInfer.h>

#include "segmentation.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"

class Logger : public nvinfer1::ILogger  
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;



SegmentationInfer::SegmentationInfer(std::string engine_path)
{ 

    std::ifstream ifile(engine_path, std::ios::in | std::ios::binary);
    if (!ifile) {
      std::cout << "read serialized file failed\n";
      std::abort();
    }

    ifile.seekg(0, std::ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    std::vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();
    std::cout << "model size: " << mdsize << std::endl;

    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime -> deserializeCudaEngine((void*)&buf[0], mdsize, nullptr);
    std::cout << "Load Success" << std::endl;       

    context = engine->createExecutionContext();

    // Setup I/O bindings 
    int numBindings = engine->getNbBindings();  
    std::cout << "Num Bindings : " << numBindings << std::endl; 
 
    for (int i = 0; i < numBindings; i++){ 
        const char* binding_name = engine->getBindingName(i); 
        std::string tensorName = binding_name; 

        bool is_input = false; 

        // Get the tensor mode
        nvinfer1::TensorIOMode tensorMode = engine->getTensorIOMode(tensorName.c_str());

        // Check if the tensor mode is INPUT
        if (tensorMode == nvinfer1::TensorIOMode::kINPUT) {
            is_input = true; 
        }

        dtype = engine->getBindingDataType(i); 
        shape = engine->getBindingDimensions(i);

        // Allocate GPU memory to hold the entire batch
        size_t m_inputCount = 1; 
        for(int i = 0; i<shape.nbDims; i++)
        { 
            m_inputCount *= shape.d[i]; 
        }

        cudaMalloc(&m_deviceInput, m_inputCount * sizeof(float));


    }



}