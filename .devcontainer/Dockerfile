FROM nvcr.io/nvidia/tensorrt:23.05-py3

# opencv
RUN apt-get update
RUN git clone https://github.com/opencv/opencv.git /opt/opencv
RUN git clone https://github.com/opencv/opencv_contrib.git /opt/opencv_contrib
RUN cd /opt/opencv && \ 
    mkdir build && \ 
    cd build && \ 
    cmake -D WITH_CUDA=ON \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D CUDA_ARCH_BIN=8.6 \
          -D CUDA_ARCH_PTX=8.6 \
          -D WITH_CUBLAS=1 \
           ..
