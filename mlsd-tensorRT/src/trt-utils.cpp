
#include "trt-utils.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include <unistd.h>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;   


namespace TRT {

    bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
        if(code != cudaSuccess){    
            const char* err_name = cudaGetErrorName(code);    
            const char* err_message = cudaGetErrorString(code);  
            printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
            return false;
        }
        return true;
    }


    std::vector<unsigned char> load_file(const std::string& file){
        ifstream in(file, ios::in | ios::binary);
        if (!in.is_open())
            return {};

        in.seekg(0, ios::end);
        size_t length = in.tellg();

        std::vector<uint8_t> data;
        if (length > 0){
            in.seekg(0, ios::beg);
            data.resize(length);

            in.read((char*)&data[0], length);
        }
        in.close();
        return data;
    }


	bool save_file(const std::string& file, const void* data, size_t length){

        FILE* f = fopen(file.c_str(), "wb");
        if (!f) return false;

        if (data && length > 0){
            if (fwrite(data, 1, length, f) not_eq length){
                fclose(f);
                return false;
            }
        }
        fclose(f);
        return true;
    }


	template<typename _T>
	shared_ptr<_T> make_nvshared(_T* ptr){
		return shared_ptr<_T>(ptr, [](_T* p){ p->destroy();});
	}



    bool exists(const std::string& path){

    #ifdef _WIN32
        return ::PathFileExistsA(path.c_str());
    #else
        return access(path.c_str(), R_OK) == 0;
    #endif
    }


	bool compile(const string& source, const string& saveto, unsigned int maxBatchSize, const size_t maxWorkspaceSize, bool useFP16) {
        if(exists(saveto)){
            printf("Engine.trtmodel has exists.\n");
            return true;
        }
        TRTLogger logger;
		auto builder = make_nvshared(createInferBuilder(logger));
		if (builder == nullptr) {
			printf("Can not create builder.\n");
			return false;
		}

		auto config = make_nvshared(builder->createBuilderConfig());
		if (useFP16) {
			if (!builder->platformHasFastFp16()) {
				printf("Platform not have fast fp16 support.\n");
			}
			config->setFlag(BuilderFlag::kFP16);
		}

		shared_ptr<INetworkDefinition> network;
		const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
		network = make_nvshared(builder->createNetworkV2(explicitBatch));
		
		shared_ptr<nvonnxparser::IParser> onnxParser = make_nvshared(nvonnxparser::createParser(*network, logger));
		if (onnxParser == nullptr) {
			printf("Can not create parser.\n");
			return false;
		}

		if (!onnxParser->parseFromFile(source.c_str(), 1)) {
			printf("Can not parse OnnX file: %s\n", source.c_str());
			return false;
		}

		auto inputTensor = network->getInput(0);
		auto inputDims = inputTensor->getDimensions();

		printf("Set max batch size = %d\n", maxBatchSize);
		printf("Set max workspace size = %.2f MB\n", maxWorkspaceSize / 1024.0f / 1024.0f);

		int net_num_input = network->getNbInputs();
		printf("Network has %d inputs.\n", net_num_input);

		int net_num_output = network->getNbOutputs();
		printf("Network has %d outputs.\n", net_num_output);
	
		builder->setMaxBatchSize(maxBatchSize);
		config->setMaxWorkspaceSize(maxWorkspaceSize);

		auto profile = builder->createOptimizationProfile();
		for(int i = 0; i < net_num_input; ++i){
			auto input = network->getInput(i);
			auto input_dims = input->getDimensions();
			input_dims.d[0] = 1;
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
			input_dims.d[0] = maxBatchSize;
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
		}

		config->addOptimizationProfile(profile);

		printf("Building engine...\n");
		auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
		if (engine == nullptr) {
			printf("engine is nullptr.\n");
			return false;
		}
		printf("Build done.\n");

		auto seridata = make_nvshared(engine->serialize());
		return save_file(saveto, seridata->data(), seridata->size());
	}


    void EngineContext::set_stream(cudaStream_t stream){

        if(owner_stream_){
            if (stream_) {cudaStreamDestroy(stream_);}
            owner_stream_ = false;
        }
        stream_ = stream;
    }


    bool EngineContext::build_model(const void* pdata, size_t size) {
        destroy();

        if(pdata == nullptr || size == 0)
            return false;

        owner_stream_ = true;
        checkRuntime(cudaStreamCreate(&stream_));
        if(stream_ == nullptr)
            return false;

        runtime_ = make_nvshared(createInferRuntime(logger));
        if (runtime_ == nullptr)
            return false;

        engine_ = make_nvshared(runtime_->deserializeCudaEngine(pdata, size, nullptr));
        if (engine_ == nullptr)
            return false;

        //runtime_->setDLACore(0);
        context_ = make_nvshared(engine_->createExecutionContext());
        return context_ != nullptr;
    }


    void EngineContext::destroy() {
        context_.reset();
        engine_.reset();
        runtime_.reset();

        if(owner_stream_){
            if (stream_) {cudaStreamDestroy(stream_);}
        }
        stream_ = nullptr;
    }


    std::shared_ptr<TRT::EngineContext> create_context(const std::string& engineFile){
        auto data = load_file(engineFile);
        if (data.empty())
            return nullptr;

        std::shared_ptr<TRT::EngineContext> engine_context;
        engine_context.reset(new TRT::EngineContext());
        //build model
        if (!engine_context->build_model(data.data(), data.size())) {
            engine_context.reset();
            printf("Deserialize cuda engine failed.\n");
            return nullptr;
        }

        cudaStream_t stream = nullptr;
        engine_context->set_stream(stream);

        // if(engine_context->engine_->getNbBindings() != 3){
        //     printf("你的onnx导出有问题,必须是1个输入和2个输出,你这明显有：%d个输出.\n", engine_context->engine_->getNbBindings() - 1);
        //     return nullptr;
        // }
        return engine_context;
    }


}; //namespace TRTBuilder
