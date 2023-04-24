cmake_minimum_required(VERSION 3.10)
project(mlsd-trt)

# 设置用debug还是release模式。debug允许断点，而release更快
# set( CMAKE_BUILD_TYPE Debug)
set( CMAKE_BUILD_TYPE Release)

# 设置编译选项
# 允许c++11标准、O3优化、多线程。march选项可避免一些CPU上的问题
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread")

set(cuda_home "/home/cyh/CodeFile/CPP/trt8cuda112cudnn8")

set(OpenCV_DIR   "/home/cyh/opencv455/lib/cmake/opencv4/")
find_package(OpenCV REQUIRED)
# 头文件目录
INCLUDE_DIRECTORIES(
    ./src
    /home/cyh/CodeFile/CPP/trt8cuda112cudnn8/include/cuda
    /home/cyh/CodeFile/CPP/trt8cuda112cudnn8/include/tensorRT
    # ${PROJECT_SOURCE_DIR}/trt8cuda112cudnn8/include/protobuf
    /home/cyh/opencv455/include/opencv4
)

FILE(GLOB SRC_LIST_CPP
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_executable(${PROJECT_NAME} ${SRC_LIST_CPP})
target_link_libraries(${PROJECT_NAME}
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_dnn
    /home/cyh/CodeFile/CPP/trt8cuda112cudnn8/lib64/libcudart.so
    /home/cyh/CodeFile/CPP/trt8cuda112cudnn8/lib64/libcudnn.so
    /home/cyh/CodeFile/CPP/trt8cuda112cudnn8/lib64/libnvinfer.so
    /home/cyh/CodeFile/CPP/trt8cuda112cudnn8/lib64/libnvonnxparser.so
)


