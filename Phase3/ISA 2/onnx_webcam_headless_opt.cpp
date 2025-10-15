#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

// Shared data
Mat latest_frame;
mutex frame_mutex;
atomic<bool> running(true);
atomic<bool> motion_detected(false);

// Motion detection parameters
const double MOTION_THRESHOLD = 20.0;  // sensitivity
const int MIN_MOTION_PIXELS = 5000;    // number of changed pixels to trigger inference

// Helper for intersection over union (IoU)
float IoU(const Rect& a, const Rect& b) {
    int x1 = max(a.x, b.x);
    int y1 = max(a.y, b.y);
    int x2 = min(a.x + a.width, b.x + b.width);
    int y2 = min(a.y + a.height, b.y + b.height);
    int interArea = max(0, x2 - x1) * max(0, y2 - y1);
    return float(interArea) / float(a.area() + b.area() - interArea);
}

// Non-maximum suppression
vector<Rect> nonMaxSuppression(const vector<Rect>& boxes, float iouThreshold) {
    vector<Rect> result;
    vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j)
            if (IoU(boxes[i], boxes[j]) > iouThreshold)
                suppressed[j] = true;
    }
    return result;
}

// Camera capture thread
void captureThread() {
    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(CAP_PROP_FPS, 10);

    if (!cap.isOpened()) {
        cerr << "Camera not found!\n";
        running = false;
        return;
    }

    Mat prev_gray, gray, diff;
    while (running) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        if (!prev_gray.empty()) {
            absdiff(gray, prev_gray, diff);
            threshold(diff, diff, MOTION_THRESHOLD, 255, THRESH_BINARY);
            int changedPixels = countNonZero(diff);
            motion_detected = (changedPixels > MIN_MOTION_PIXELS);
            
            if (motion_detected) {
                lock_guard<mutex> lock(frame_mutex);
                latest_frame = frame.clone();
            }
        }
        gray.copyTo(prev_gray);
        this_thread::sleep_for(chrono::milliseconds(30));
    }
    cap.release();
}

// Inference thread
void inferenceThread() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    Ort::Session session(env, "model.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = { input_name.get() };
    const char* output_names[] = { output_name.get() };

    while (running) {
        if (!motion_detected) {
            this_thread::sleep_for(chrono::milliseconds(500));
            continue;
        }

        Mat frame;
        {
            lock_guard<mutex> lock(frame_mutex);
            if (latest_frame.empty()) continue;
            frame = latest_frame.clone();
        }

        // Preprocess frame
        Mat resized;
        resize(frame, resized, Size(640, 640));
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);

        vector<float> input_tensor_values;
        input_tensor_values.reserve(3 * 640 * 640);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < 640; ++y)
                for (int x = 0; x < 640; ++x)
                    input_tensor_values.push_back(resized.at<Vec3f>(y, x)[c]);

        vector<int64_t> input_shape = {1, 3, 640, 640};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

        // Run model
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        float* output_data = output_tensors.front().GetTensorMutableData<float>();

        // Example dummy filtering (depends on model)
        vector<Rect> boxes;
        for (int i = 0; i < 100; ++i) {
            int x = int(output_data[i * 6 + 0]);
            int y = int(output_data[i * 6 + 1]);
            int w = int(output_data[i * 6 + 2]);
            int h = int(output_data[i * 6 + 3]);
            float conf = output_data[i * 6 + 4];
            if (conf > 0.5)
                boxes.emplace_back(x, y, w, h);
        }

        boxes = nonMaxSuppression(boxes, 0.45);

        cout << "Inference done | " << boxes.size() << " boxes\n";
        motion_detected = false;  // reset

        this_thread::sleep_for(chrono::milliseconds(50));
    }
}

int main() {
    thread t1(captureThread);
    thread t2(inferenceThread);

    cout << "Running headless ONNX inference (press Ctrl+C to stop)...\n";
    while (running) {
        this_thread::sleep_for(chrono::seconds(1));
    }

    t1.join();
    t2.join();
    return 0;
}
