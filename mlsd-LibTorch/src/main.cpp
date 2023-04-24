#include <mlsd.h>
#include <unistd.h>

using namespace std;


void demo_mlsd(){
    cv::Mat frame = cv::imread("/home/cyh/CodeFile/CPP/mlsd-inference/mlsd-LibTorch/workspace/frame.jpg");
    std::string ptFile = "/home/cyh/CodeFile/CPP/mlsd-inference/workspace/mlsd-large.cpu.torchscript";
    std::string savepath = "/home/cyh/CodeFile/CPP/mlsd-inference/workspace/frame-res.jpg";
    MLsd detector(ptFile, false, 0.2f, 2.0f);
    detector.inference(frame, savepath);
}

int main(){

    sleep(5);
    demo_mlsd();

    sleep(10);

    return 0;
}