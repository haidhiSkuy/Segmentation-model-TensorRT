#include "src/segmentation.h"
#include <opencv2/opencv.hpp> 
#include <iostream>
#include "NvInfer.h" 




int main()
{ 
    SegmentationInfer segmen("/workspaces/tensorRT/model/ghost_unet.engine");

    
    return 0; 
}