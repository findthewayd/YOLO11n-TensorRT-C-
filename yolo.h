#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp> // 必须包含，用于 NMSBoxes
#include "NvInfer.h"

using namespace nvinfer1;

struct DetectResult {
    cv::Rect rect;
    int class_id;
    float confidence;
};

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

class YOLO {
public:
    YOLO();
    ~YOLO();
    bool loadEngine(const std::string& engine_path);
    std::vector<DetectResult> infer(cv::Mat& frame);

private:
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    void* device_input = nullptr;
    void* device_output = nullptr;
    float* host_input = nullptr;
    float* host_output = nullptr;

    cudaStream_t cuda_stream = nullptr;

    void preprocess(cv::Mat& frame, float* input, int width, int height);
    std::vector<DetectResult> postprocess(float* output, int width, int height, float conf_thresh = 0.45f);
};