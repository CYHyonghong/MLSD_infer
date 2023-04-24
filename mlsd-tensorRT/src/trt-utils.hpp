
#ifndef TRT_UTILS_HPP
#define TRT_UTILS_HPP

#include <string>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

#define checkRuntime(op) TRT::__check_cuda_runtime((op), #op, __FILE__, __LINE__)


namespace TRT {

    bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

    class TRTLogger : public nvinfer1::ILogger{
    public:
        virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
            if(severity <= Severity::kINFO){
                printf("%d: %s\n", severity, msg);
            }
        }
    };

    bool exists(const std::string& path);

	bool compile(
		const std::string& source,
		const std::string& saveto,
        unsigned int maxBatchSize = 1, 
		const size_t maxWorkspaceSize = 1ul << 30,     // 1ul << 30 = 1GB
        bool useFP16 = false
	);

    std::vector<unsigned char> load_file(const std::string& file);


    class EngineContext {
    public:
        virtual ~EngineContext() { destroy(); }

        void set_stream(cudaStream_t stream);

        bool build_model(const void* pdata, size_t size);

    private:
        void destroy();

    public:
        TRTLogger logger;
        cudaStream_t stream_ = nullptr;
        bool owner_stream_ = false;
        std::shared_ptr<nvinfer1::IExecutionContext> context_;
        std::shared_ptr<nvinfer1::ICudaEngine> engine_;
        std::shared_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
    };

    std::shared_ptr<TRT::EngineContext> create_context(const std::string& engineFile);

};

#endif //TRT_UTILS_HPP