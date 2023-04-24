#ifndef MLSD_H
#define MLSD_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>


struct Sort_st{
    int index;
    float value;
};


class MLsd{

public:
    MLsd(std::string& ptFile, bool isCuda, float scoreThres, float lenThres);

    std::vector<std::vector<float>> inference(cv::Mat& frame, std::string& savePath);

private:
    float score_threshold;
    float len_threshold;
    int input_height = 512;
    int input_width  = 512;
    torch::jit::script::Module module_;
    torch::Device device_ = torch::kCPU;


};



#endif // MLSD_H