
#ifndef MLSD_TRT_HPP
#define MLSD_TRT_HPP

#include <trt-utils.hpp>
#include <opencv2/opencv.hpp>

void mlsd_inference(std::shared_ptr<TRT::EngineContext>& context, cv::Mat& image, std::vector<std::vector<float>>& lines, int input_height=512, int input_width=512);

int mlsd_demo();



#endif //!MLSD_TRT_HPP