#include "mlsd.h"
#include <math.h>
#include <algorithm>

using namespace std;
using namespace cv;


MLsd::MLsd(std::string& ptFile, bool isCuda, float scoreThres, float lenThres){
    try{
        module_ = torch::jit::load(ptFile);
    }
    catch(const c10::Error& e){
        std::cerr << "Error loading the model\n";
        std::exit(EXIT_FAILURE);
    }
    if(isCuda){
        this->device_ = torch::kCUDA;
    }
    module_.to(this->device_);
    this->score_threshold = scoreThres;
    this->len_threshold  = lenThres;
    module_.eval();
}

std::vector<std::vector<float>> MLsd::inference(cv::Mat& frame, string& savePath){

    printf("---------- Begin preprocess ----------\n");
    /** preprocess **/
    int frame_h = frame.rows;
    int frame_w = frame.cols;
    cv::Mat image = frame.clone();
    float mean[] = {0.485, 0.456, 0.406};
    float std[]  = {0.229, 0.224, 0.225};

    int input_batch = 1;
    int input_channels = 3;
    int input_numel = input_batch * input_channels * this->input_height * this->input_width;
    float input_data[input_numel];

    cv::resize(image, image, cv::Size(this->input_width, this->input_height));
    int image_area = image.rows * image.cols;
    unsigned char* pimage = image.data;
    float* phost_b = input_data + image_area * 0;
    float* phost_g = input_data + image_area * 1;
    float* phost_r = input_data + image_area * 2;
    for(int i = 0; i < image_area; i++, pimage += 3){
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
    }

    auto input_tensor = torch::from_blob(input_data, {input_batch, input_channels, this->input_height, this->input_width}).to(this->device_);
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    /** inference **/
    printf("---------- Begin inference ----------\n");
    auto outputs = this->module_.forward(inputs).toTuple();
    auto heat = outputs->elements()[0].toTensor();
    auto displacement = outputs->elements()[1].toTensor();

    /** postprocess **/
    printf("---------- Begin postprocess ----------\n");
    // std::cout << "heat size : " << heat.sizes() << std::endl;
    // std::cout << "displacement size : " << displacement.sizes() << std::endl;
    float* heat_data = (float*)heat.cpu().data_ptr();
    float* displacement_data = (float*)displacement.cpu().data_ptr();

    int topk = 200;
    int heat_feats = 65536;
    int displacement_channels = 4;
    int displacement_numlines = 65536;

    std::vector<Sort_st> sort_array(heat_feats);
    for(int i = 0; i< heat_feats; ++i){
        sort_array[i].index = i;
        sort_array[i].value = heat_data[i];
    }
    sort(sort_array.begin(),sort_array.end(),[](const Sort_st& a, const Sort_st& b){return a.value > b.value;});

    std::vector<std::vector<float>> lines;
    for(int i = 0; i < topk; i++){
        if(sort_array[i].value > this->score_threshold){
            // printf("sort_array top k  index = %d, value = %f\n", sort_array[i].index, sort_array[i].value);
            int center_x = sort_array[i].index % 256;
            int center_y = sort_array[i].index / 256;

            float start_x = center_x + displacement_data[sort_array[i].index + 0 * displacement_numlines];
            float start_y = center_y + displacement_data[sort_array[i].index + 1 * displacement_numlines];

            float end_x = center_x + displacement_data[sort_array[i].index + 2 * displacement_numlines];
            float end_y = center_y + displacement_data[sort_array[i].index + 3 * displacement_numlines]; 

            if(sqrt(pow(end_x - start_x, 2) + pow(end_y - start_y, 2)) > this->len_threshold){
                float new_start_x   = frame_w * start_x / this->input_width * 2.0f;
                float new_start_y   = frame_h * start_y / this->input_height * 2.0f;
                float new_end_x     = frame_w * end_x / this->input_width * 2.0f;
                float new_end_y     = frame_h * end_y / this->input_height * 2.0f;
                lines.push_back({new_start_x, new_start_y, new_end_x, new_end_y});
            }
        }
        else break;
    }


    if(lines.size() != 0){
        for(int j = 0; j < lines.size(); ++j){
            cv::line(frame, cv::Point(lines[j][0], lines[j][1]), cv::Point(lines[j][2], lines[j][3]), cv::Scalar(0, 0, 255), 2);
        }
    }
    cv::imwrite(savePath, frame);

    return lines;
}


