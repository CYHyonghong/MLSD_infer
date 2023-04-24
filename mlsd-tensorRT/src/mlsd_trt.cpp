#include "mlsd_trt.hpp"
#include <cuda_runtime.h>

using namespace std;

typedef struct{
    int index;
    float value;
}sort_st;

bool compare(sort_st a,sort_st b){
    return a.value > b.value; // 降序排列
}

void mlsd_inference(std::shared_ptr<TRT::EngineContext>& context, cv::Mat& image, std::vector<std::vector<float>>& lines, int input_height, int input_width){
    if(image.empty()){
        printf("input image is empty.\n");
        return;
    }

    int input_batch = 1;
    int input_channels = 3;
    int input_numel = input_batch * input_channels * input_height * input_width;
    float* input_data_host = nullptr;
    float* input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    int image_h = image.rows;
    int image_w = image.cols;
    auto ori_image = image.clone();
    float mean[] = {0.485, 0.456, 0.406};
    float std[]  = {0.229, 0.224, 0.225};

    cv::resize(image, image, cv::Size(input_width, input_height));
    int image_area = image.rows * image.cols;
    unsigned char* pimage = image.data;
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;
    for(int i = 0; i < image_area; i++, pimage += 3){
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
    }
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, context->stream_));

    int topk = 200;
    float len_threshold = 4.0f;
    float score_threshold = 0.2f;
    auto heat_dims = context->engine_->getBindingDimensions(1);
    int heat_feats = heat_dims.d[0];
    printf("heat_channels: %d\n", heat_feats);
    auto displacement_dims = context->engine_->getBindingDimensions(2);
    int displacement_channels = displacement_dims.d[1];
    int displacement_numlines = displacement_dims.d[2];
    printf("channels: %d, numlines: %d\n", displacement_channels, displacement_numlines);

    float heat_data_host[heat_feats];
    float displacement_data_host[displacement_channels][displacement_numlines];
    float* heat_data_device = nullptr;
    float* displacement_data_device = nullptr;

    // checkRuntime(cudaMallocHost(&displacement_data_host, displacement_channels * displacement_numlines * sizeof(float)));
    checkRuntime(cudaMalloc(&heat_data_device, sizeof(heat_data_host)));
    checkRuntime(cudaMalloc(&displacement_data_device, displacement_channels * displacement_numlines * sizeof(float)));

    auto name0 = context->engine_->getBindingName(0);
    auto name1 = context->engine_->getBindingName(1);
    auto name2 = context->engine_->getBindingName(2);
    printf("name0: %s, name1: %s, name2: %s\n", name0, name1, name2);

    float* bindings[] = {input_data_device, heat_data_device, displacement_data_device};
    bool success    = context->context_->enqueueV2((void**)bindings, context->stream_, nullptr);
    checkRuntime(cudaMemcpyAsync(heat_data_host, heat_data_device, sizeof(heat_data_host), cudaMemcpyDeviceToHost, context->stream_));
    checkRuntime(cudaMemcpyAsync(displacement_data_host, displacement_data_device, displacement_channels * displacement_numlines * sizeof(float), cudaMemcpyDeviceToHost, context->stream_));
    checkRuntime(cudaStreamSynchronize(context->stream_));

    vector <sort_st> sort_array(heat_feats);
    for(int i = 0; i< heat_feats; ++i){
        sort_array[i].index = i;
        sort_array[i].value = heat_data_host[i];
    }
    sort(sort_array.begin(), sort_array.end(), compare);

    for(int i = 0; i < topk; i++){
        if(sort_array[i].value > score_threshold){
            // printf("sort_array top k  index = %d, value = %f\n", sort_array[i].index, sort_array[i].value);
            int center_x = sort_array[i].index % 256;
            int center_y = sort_array[i].index / 256;

            float start_x = center_x + displacement_data_host[0][sort_array[i].index];
            float start_y = center_y + displacement_data_host[1][sort_array[i].index];

            float end_x = center_x + displacement_data_host[2][sort_array[i].index];
            float end_y = center_y + displacement_data_host[3][sort_array[i].index]; 

            if(sqrt(pow(end_x - start_x, 2) + pow(end_y - start_y, 2)) > len_threshold){
                float new_start_x   = image_w * start_x / input_width * 2.0f;
                float new_start_y   = image_h * start_y / input_height * 2.0f;
                float new_end_x     = image_w * end_x / input_width * 2.0f;
                float new_end_y     = image_h * end_y / input_height * 2.0f;
                lines.push_back({new_start_x, new_start_y, new_end_x, new_end_y});
            }
 
        }
    }

    checkRuntime(cudaFree(displacement_data_device));
    checkRuntime(cudaFree(heat_data_device));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
    printf("infer end.\n");

}

int mlsd_demo(){
    std::string onnxFile = "/home/cyh/CodeFile/CPP/mlsd-inference/mlsd-tensorRT/workspace/mlsd-pointer.onnx";
    std::string engineFile = "/home/cyh/CodeFile/CPP/mlsd-inference/mlsd-tensorRT/workspace/mlsd-pointer.trtmodel";
    std::string imageFile = "/home/cyh/CodeFile/CPP/mlsd-inference/mlsd-tensorRT/workspace/frame.jpg";
    std::string savepath = "/home/cyh/CodeFile/CPP/mlsd-inference/mlsd-tensorRT/workspace/frame-res.jpg";

    if(!TRT::compile(onnxFile, engineFile, 1, 1 << 28)){
        return -1;
    }
    // create execution_context
    std::shared_ptr<TRT::EngineContext> context = TRT::create_context(engineFile);
    if(context == nullptr){
        return -1;
    }

    cv::Mat image = cv::imread(imageFile);
    cv::Mat ori_image = image.clone();
    std::vector<std::vector<float>> lines;

    for(int i = 0; i < 1; i++){
        lines.clear();
        auto start = chrono::steady_clock::now();
        mlsd_inference(context, image, lines);
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