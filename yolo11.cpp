#include "yolo.h"

Logger gLogger;

YOLO::YOLO() {}

YOLO::~YOLO() {
    if (cuda_stream) cudaStreamDestroy(cuda_stream);
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
    if (device_input) cudaFree(device_input);
    if (device_output) cudaFree(device_output);
    if (host_input) delete[] host_input;
    if (host_output) delete[] host_output;
}

bool YOLO::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) return false;

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    runtime = createInferRuntime(gLogger);
    if (!runtime) return false;
    engine = runtime->deserializeCudaEngine(engine_data.data(), size);
    if (!engine) return false;
    context = engine->createExecutionContext();
    if (!context) return false;

    cudaStreamCreate(&cuda_stream);

    host_input = new float[1 * 3 * 640 * 640];
    host_output = new float[1 * 5 * 8400];
    cudaMalloc(&device_input, 1 * 3 * 640 * 640 * sizeof(float));
    cudaMalloc(&device_output, 1 * 5 * 8400 * sizeof(float));

    context->setTensorAddress(engine->getIOTensorName(0), device_input);
    context->setTensorAddress(engine->getIOTensorName(1), device_output);

    return true;
}

void YOLO::preprocess(cv::Mat& frame, float* input, int w, int h) {
    cv::Mat rgb, resized;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(w, h));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    int area = w * h;
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = cv::Mat(h, w, CV_32FC1, input + i * area);
    }
    cv::split(resized, channels);
}

std::vector<DetectResult> YOLO::infer(cv::Mat& frame) {
    preprocess(frame, host_input, 640, 640);
    cudaMemcpyAsync(device_input, host_input, 1 * 3 * 640 * 640 * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
    context->enqueueV3(cuda_stream);
    cudaMemcpyAsync(host_output, device_output, 1 * 5 * 8400 * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);

    return postprocess(host_output, frame.cols, frame.rows, 0.25f);
}

std::vector<DetectResult> YOLO::postprocess(float* output, int img_w, int img_h, float conf_thresh) {
    std::vector<cv::Rect> bboxes;
    std::vector<float> confs;
    std::vector<int> class_ids;

    // 重点：YOLO11 导出后通常是 [5, 8400] 格式，需跨步长读取
    for (int i = 0; i < 8400; ++i) {
        float conf = output[i + 4 * 8400];
        if (conf < conf_thresh) continue;

        float cx = output[i + 0 * 8400] * img_w / 640.0f;
        float cy = output[i + 1 * 8400] * img_h / 640.0f;
        float w = output[i + 2 * 8400] * img_w / 640.0f;
        float h = output[i + 3 * 8400] * img_h / 640.0f;

        bboxes.push_back(cv::Rect(int(cx - w / 2), int(cy - h / 2), int(w), int(h)));
        confs.push_back(conf);
        class_ids.push_back(0);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, confs, conf_thresh, 0.45f, indices);

    std::vector<DetectResult> results;
    for (int idx : indices) {
        results.push_back({ bboxes[idx], class_ids[idx], confs[idx] });
    }
    return results;
}