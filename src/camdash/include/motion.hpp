#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class MotionDetector {
public:
    MotionDetector(int width, int height, double thresh = 25.0, double minChange = 0.02);
    bool detect(const std::vector<unsigned char>& jpegFrame);

private:
    cv::Mat prevGray_;
    int width_, height_;
    double thresh_;
    double minChange_;
};

