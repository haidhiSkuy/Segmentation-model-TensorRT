#include <string> 
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

class SegmentationInfer
{ 
    public: 
        SegmentationInfer(std::string engine_path);

    private: 
        // define model engine
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::IExecutionContext* context = nullptr; 

        // define model input and output shape 
        nvinfer1::DataType dtype; 
        nvinfer1::Dims shape; 

        void *m_deviceInput; 
        void *m_deviceOutput;
};



