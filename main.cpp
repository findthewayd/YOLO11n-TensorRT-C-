#include "yolo.h"
#include <chrono>

int main() {
    YOLO yolo;
    // 强制使用无空格的绝对路径以规避 image_2d7ff4.png 中的加载失败
    if (!yolo.loadEngine("D:\\TensorRT1016\\bin\\bestuav.engine")) {
        std::cerr << "Engine load failed! Check your path." << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    cv::Mat frame;
    while (cap.read(frame)) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto results = yolo.infer(frame);
        auto t2 = std::chrono::high_resolution_clock::now();

        float fps = 1000.0f / std::chrono::duration<float, std::milli>(t2 - t1).count();

        for (auto& res : results) {
            cv::rectangle(frame, res.rect, cv::Scalar(0, 255, 0), 2);
            std::string text = "UAV: " + std::to_string(res.confidence).substr(0, 4);
            cv::putText(frame, text, cv::Point(res.rect.x, res.rect.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        cv::putText(frame, "FPS: " + std::to_string(fps).substr(0, 4), cv::Point(30, 50),
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Anti-UAV System", frame);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}