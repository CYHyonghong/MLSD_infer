#include "mlsd-dnn.h"
#include <stdlib.h>
#include <vector>

using namespace std;
using namespace cv;

typedef struct{
    int index;
    float value;
}Sort_St;

MLSD::MLSD(string modelFile, float scoreThres, float lenThres, int height, int width){
    input_height_ = height;
    input_width_;
    score_threshold_ = scoreThres;
    len_threshold_  = lenThres;
    try{
        net_ = cv::dnn::readNet(modelFile);
    }
    catch(cv::Exception& e){
        std::cerr << "Error loading the model\n";
        std::exit(EXIT_FAILURE);
    }
}

MLSD::~MLSD(){

}

void MLSD::normalize(cv::Mat& img)
{
	img.convertTo(img, CV_32F);
	int i = 0, j = 0;
	for (i = 0; i < img.rows; i++)
	{
		float* pdata = (float*)(img.data + i * img.step);
		for (j = 0; j < img.cols; j++)
		{
			pdata[0] = (pdata[0] - mean_[0]) / std_[0];
			pdata[1] = (pdata[1] - mean_[1]) / std_[1];
			pdata[2] = (pdata[2] - mean_[2]) / std_[2];
			pdata += 3;
		}
	}
}

void MLSD::inference(cv::Mat& frame, vector<vector<float>> lines){
    printf("---------- Begin preprocess ----------\n");
    /** preprocess **/
    int frame_h = frame.rows;
    int frame_w = frame.cols;
    cv::Mat image = frame.clone();
    normalize(image);
    cv::cvtColor(image, image, COLOR_BGR2RGB);
	cv::Mat blob = cv::dnn::blobFromImage(image);
	net_.setInput(blob);

	vector<cv::Mat> outs;
	net_.forward(outs, net_.getUnconnectedOutLayersNames());
    auto heat_data = outs[0].data;
    auto displacement_data = outs[1].data;

    int topk = 200;
    int heat_feats = 65536;
    int displacement_channels = 4;
    int displacement_numlines = 65536;

    std::vector<Sort_St> sort_array(heat_feats);
    for(int i = 0; i< heat_feats; ++i){
        sort_array[i].index = i;
        sort_array[i].value = heat_data[i];
    }
    sort(sort_array.begin(),sort_array.end(),[](const Sort_St& a, const Sort_St& b){return a.value > b.value;});

    for(int i = 0; i < topk; i++){
        if(sort_array[i].value > this->score_threshold_){
            // printf("sort_array top k  index = %d, value = %f\n", sort_array[i].index, sort_array[i].value);
            int center_x = sort_array[i].index % 256;
            int center_y = sort_array[i].index / 256;

            float start_x = center_x + displacement_data[sort_array[i].index + 0 * displacement_numlines];
            float start_y = center_y + displacement_data[sort_array[i].index + 1 * displacement_numlines];

            float end_x = center_x + displacement_data[sort_array[i].index + 2 * displacement_numlines];
            float end_y = center_y + displacement_data[sort_array[i].index + 3 * displacement_numlines]; 

            if(sqrt(pow(end_x - start_x, 2) + pow(end_y - start_y, 2)) > this->len_threshold_){
                float new_start_x   = frame_w * start_x / this->input_width_ * 2.0f;
                float new_start_y   = frame_h * start_y / this->input_height_ * 2.0f;
                float new_end_x     = frame_w * end_x / this->input_width_ * 2.0f;
                float new_end_y     = frame_h * end_y / this->input_height_ * 2.0f;
                lines.push_back({new_start_x, new_start_y, new_end_x, new_end_y});
            }
        }
        else break;
    }

}

int mlsd_dnn_demo(){
    // std::string onnxFile = "/home/cyh/CodeFile/CPP/mlsd-inference/workspace/mlsd-pointer.onnx";
    std::string onnxFile = "/home/cyh/mlsd-pointer.onnx";
    std::string imageFile = "/home/cyh/CodeFile/CPP/mlsd-inference/workspace/frame.jpg";
    std::string savepath = "/home/cyh/CodeFile/CPP/mlsd-inference/workspace/frame-dnn.jpg";

    cv::Mat image = cv::imread(imageFile);
    cv::Mat ori_image = image.clone();
    std::vector<std::vector<float>> lines;

    MLSD line_detector(onnxFile, 0.2f, 2.0f);

    for(int i = 0; i < 1; i++){
        lines.clear();
        auto start = chrono::steady_clock::now();
        line_detector.inference(image, lines);
        auto end = chrono::steady_clock::now();
        chrono::duration<float> d = end - start;
        std::cout << "mlsd inference takes time : " << d.count() * 1000 << "ms.\n";
    }

    if(lines.size() != 0){
        for(int j = 0; j < lines.size(); ++j){

            cv::line(ori_image, cv::Point(lines[j][0], lines[j][1]), cv::Point(lines[j][2], lines[j][3]), cv::Scalar(0, 0, 255), 2);
        }
    }
    cv::imwrite(savepath, ori_image);

}