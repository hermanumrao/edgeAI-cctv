#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

struct Box {
    int x1, y1, x2, y2;
    float score;
    int class_id;
};

// Compute Intersection over Union
float IoU(const Box& a, const Box& b) {
    int x1 = std::max(a.x1, b.x1);
    int y1 = std::max(a.y1, b.y1);
    int x2 = std::min(a.x2, b.x2);
    int y2 = std::min(a.y2, b.y2);

    int w = std::max(0, x2 - x1);
    int h = std::max(0, y2 - y1);
    float inter = w * h;
    float union_area = (a.x2 - a.x1) * (a.y2 - a.y1) +
                       (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return inter / union_area;
}

// Non-Maximum Suppression
std::vector<Box> NMS(const std::vector<Box>& boxes, float iou_threshold) {
    std::vector<Box> result;
    std::vector<Box> sorted_boxes = boxes;

    std::sort(sorted_boxes.begin(), sorted_boxes.end(),
              [](const Box& a, const Box& b) { return a.score > b.score; });

    std::vector<bool> removed(sorted_boxes.size(), false);

    for (size_t i = 0; i < sorted_boxes.size(); ++i) {
        if (removed[i]) continue;
        result.push_back(sorted_boxes[i]);
        for (size_t j = i + 1; j < sorted_boxes.size(); ++j) {
            if (!removed[j] && IoU(sorted_boxes[i], sorted_boxes[j]) > iou_threshold) {
                removed[j] = true;
            }
        }
    }
    return result;
}

// Preprocess image to match model input
std::vector<float> preprocess(const cv::Mat& img, int width, int height) {
    cv::Mat resized, rgb;
    cv::resize(img, resized, cv::Size(width, height));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<float> input_tensor(width * height * 3);
    int channels = 3;
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                input_tensor[c * width * height + i * width + j] =
                    rgb.at<cv::Vec3f>(i, j)[c];
            }
        }
    }
    return input_tensor;
}

// Decode raw output tensor with scaling and NMS
std::vector<Box> decode_output(float* output_data, float conf_threshold, int forced_class,
                               int orig_width, int orig_height, int model_width, int model_height) {
    std::vector<Box> boxes;
    int num_boxes = 8400;

    float x_scale = static_cast<float>(orig_width) / model_width;
    float y_scale = static_cast<float>(orig_height) / model_height;

    for (int i = 0; i < num_boxes; i++) {
        float x_c = output_data[i];
        float y_c = output_data[i + num_boxes];
        float w = output_data[i + 2 * num_boxes];
        float h = output_data[i + 3 * num_boxes];
        float conf = output_data[i + 4 * num_boxes];

        if (conf > conf_threshold) {
            int x1 = static_cast<int>((x_c - w / 2) * x_scale);
            int y1 = static_cast<int>((y_c - h / 2) * y_scale);
            int x2 = static_cast<int>((x_c + w / 2) * x_scale);
            int y2 = static_cast<int>((y_c + h / 2) * y_scale);

            boxes.push_back({x1, y1, x2, y2, conf, forced_class});
        }
    }

    float iou_threshold = 0.5f; // Adjustable
    return NMS(boxes, iou_threshold);
}

// Draw bounding boxes
void draw_boxes(cv::Mat& img, const std::vector<Box>& boxes, float conf_threshold) {
    for (const auto& box : boxes) {
        if (box.score < conf_threshold) continue;

        cv::rectangle(img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2),
                      cv::Scalar(0, 255, 0), 2);
        cv::putText(img, "Class:" + std::to_string(box.class_id) + " " +
                              std::to_string(box.score),
                    cv::Point(box.x1, box.y1 - 10), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 255, 0), 2);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <image.jpg>\n";
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Image not found!\n";
        return -1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);

    const char* input_name_c = input_name.get();
    const char* output_name_c = output_name.get();

    std::vector<const char*> input_names = {input_name_c};
    std::vector<const char*> output_names = {output_name_c};

    int model_width = 640;
    int model_height = 640;

    std::vector<float> input_tensor = preprocess(img, model_width, model_height);
    std::vector<int64_t> input_shape = {1, 3, model_height, model_width};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor.size(),
        input_shape.data(), input_shape.size());

    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names.data(), &input_tensor_ort, 1,
                                      output_names.data(), 1);

    auto& output_tensor = output_tensors.front();
    float* output_data = output_tensor.GetTensorMutableData<float>();

    float conf_threshold = 0.5f;
    auto boxes = decode_output(output_data, conf_threshold, 0,
                               img.cols, img.rows, model_width, model_height);

    draw_boxes(img, boxes, conf_threshold);

    // Bigger window
    cv::namedWindow("Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("Detection", 1280, 720);
    cv::imshow("Detection", img);

    cv::waitKey(0);
    return 0;
}
