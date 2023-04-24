
#ifndef MLSD_DNN_H
#define MLSD_DNN_H

#include <string>
#include <opencv2/opencv.hpp>

class MLSD{
public:
    MLSD(std::string modelFile, float scoreThres, float lenThres, int height=512, int width=512);
    ~MLSD();
    void inference(cv::Mat& frame, std::vector<std::vector<float>> lines);

private:

    void normalize(cv::Mat& img);

    cv::dnn::Net net_;
    float score_threshold_;
    float len_threshold_;
    int input_height_;
    int input_width_;

    // const float mean_[3] = {0.485, 0.456, 0.406};
    // const float std_[3]  = {0.229, 0.224, 0.225};
    const float mean_[3] = {123.675f, 116.28f, 103.53f};
    const float std_[3]  = {58.395f, 57.12f, 57.375f};

};


int mlsd_dnn_demo();


#endif //!MLSD_DNN_H