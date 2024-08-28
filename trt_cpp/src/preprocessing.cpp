#include "segmentation.h" 
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>


void SegmentationInfer::preprocessing(std::string image_path)
{ 
    cv::Mat cpuImage = cv::imread(image_path); 
    if (cpuImage.empty()){ 
        std::cout << "Read Image Error" << std::endl; 
    } else { 
        std::cout << "Read Image Success" << std::endl; 
    }

    // upload image to CUDA 
    cv::cuda::GpuMat GpuImage; 
    GpuImage.upload(cpuImage);  

    // convert to RGB 
    cv::cuda::cvtColor(GpuImage, GpuImage, cv::COLOR_BGR2RGB);

    // Resize
    int width = shape.d [2]; 
    int height = shape.d[3]; 
    cv::Size newSize(width, height);
    cv::cuda::resize(GpuImage, GpuImage, newSize); 
    
    // normalize
    GpuImage.convertTo(GpuImage, CV_32F, 1.0 / 255, 0);

    // copy to GPU
    auto *dataPointer = GpuImage.ptr<void>();

    cudaStream_t cudaStream;
    cudaStreamCreate(&cudaStream);
    cudaStreamSynchronize(cudaStream);
    cudaMemcpyAsync(m_deviceInput, dataPointer, m_inputCount * sizeof(float), cudaMemcpyDeviceToDevice, cudaStream); 

    context->enqueueV2(&m_deviceInput, cudaStream, nullptr);

}