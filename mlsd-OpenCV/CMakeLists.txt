cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(mlsd-dnn)

# 如果你的opencv找不到，可以自己指定目录
set(OpenCV_DIR   "/home/cyh/opencv455/lib/cmake/opencv4/")
find_package(OpenCV REQUIRED)
include_directories(
    /home/cyh/opencv455/include/opencv4
    ${PROJECT_SOURCE_DIR}/include
)

FILE(GLOB SRC_LIST_CPP
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_executable(${PROJECT_NAME} ${SRC_LIST_CPP})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES}
    opencv_core    
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
    opencv_dnn
)
