#include "src/segmentation.h"
#include <opencv2/opencv.hpp> 
#include <iostream>
#include "NvInfer.h" 


int main()
{ 
    SegmentationInfer segmen("/workspaces/tensorRT/model/ghost_unet.engine");
    
    std::string image_path = "/workspaces/tensorRT/sample_image/image4.png"; 
    segmen.preprocessing(image_path);  
    
    return 0; 
}