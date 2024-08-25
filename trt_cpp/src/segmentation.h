#include <string> 
#include <NvInfer.h>

class SegmentationInfer
{ 
    public: 
        SegmentationInfer(std::string engine_path);

    private: 
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;


};



