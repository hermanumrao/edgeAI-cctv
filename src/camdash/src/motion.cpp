#include "motion.hpp"

MotionDetector::MotionDetector(int width, int height, double thresh, double minChange)
    : width_(width), height_(height), thresh_(thresh), minChange_(minChange) {}

bool MotionDetector::detect(const std::vector<unsigned char>& jpegFrame) {
    // Decode JPEG to color
    cv::Mat img = cv::imdecode(jpegFrame, cv::IMREAD_COLOR);
    if (img.empty()) return false;

    // Resize for performance if needed
    cv::resize(img, img, cv::Size(width_, height_));

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    if (prevGray_.empty()) {
        prevGray_ = gray.clone();
        return false;
    }

    // Compute absolute difference
    cv::Mat diff;
    cv::absdiff(gray, prevGray_, diff);

    // Threshold
    cv::threshold(diff, diff, thresh_, 255, cv::THRESH_BINARY);

    // Count changed pixels
    double changeRatio = (cv::countNonZero(diff) * 1.0) / (diff.rows * diff.cols);

    // Save current as previous
    prevGray_ = gray.clone();

    // Detect motion
    if (changeRatio > minChange_) {
        return true;
    }
    return false;
}

