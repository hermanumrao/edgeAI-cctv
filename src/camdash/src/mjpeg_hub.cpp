#include "mjpeg_hub.hpp"
#include "config.hpp"
#include "utils.hpp"
#include "motion.hpp"          // <-- MotionDetector
#include <cstdio>
#include <cstring>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

MjpegHub::MjpegHub()
    : width_(cfg::CAM_WIDTH), height_(cfg::CAM_HEIGHT), fps_(cfg::CAM_FPS) {
    segment_seconds_ = cfg::SEGMENT_MINUTES * 60;

    // Start motion worker thread (runs until destructor)
    motion_running_ = true;
    motion_thread_ = std::thread(&MjpegHub::motion_thread_fn, this);
}

MjpegHub::~MjpegHub() {
    // Stop motion worker
    {
        std::lock_guard<std::mutex> lk(motion_mtx_);
        motion_running_ = false;
    }
    motion_cv_.notify_all();
    if (motion_thread_.joinable()) motion_thread_.join();

    // Stop capture/recording thread
    stop();
}

bool MjpegHub::start(int w, int h, int fps) {
    width_ = w; height_ = h; fps_ = fps;
    if (!ensure_dir(cfg::RECORD_DIR)) {
        fprintf(stderr, "Failed to create recordings dir: %s\n", cfg::RECORD_DIR.c_str());
        return false;
    }
    run_ = true;
    th_ = std::thread(&MjpegHub::reader_thread_fn, this);
    return true;
}

void MjpegHub::stop() {
    run_ = false;
    if (th_.joinable()) th_.join();
    close_segment();
}

int MjpegHub::add_sink(FrameSink cb) {
    std::lock_guard<std::mutex> lk(sinks_m_);
    int id = next_sink_id_++;
    sinks_.push_back({id, std::move(cb)});
    return id;
}

void MjpegHub::remove_sink(int id) {
    std::lock_guard<std::mutex> lk(sinks_m_);
    for (auto it = sinks_.begin(); it != sinks_.end(); ++it) {
        if (it->first == id) { sinks_.erase(it); break; }
    }
}

void MjpegHub::open_new_segment() {
    close_segment();
    std::string fname = cfg::RECORD_DIR + "/rec_" + now_timestamp_filename() + ".mjpg";
    rec_path_ = fname;
    rec_file_ = fopen(fname.c_str(), "wb");
    seconds_in_segment_ = 0;
    frames_in_segment_ = 0;
    if (!rec_file_) perror("fopen segment");
}

void MjpegHub::close_segment() {
    if (rec_file_) { fclose(rec_file_); rec_file_ = nullptr; }
}

void MjpegHub::rotate_recording_if_needed(const std::vector<unsigned char>& frame) {
    if (!rec_file_) open_new_segment();

    // Write [uint32 length][JPEG bytes]
    uint32_t len = (uint32_t)frame.size();
    fwrite(&len, 1, sizeof(len), rec_file_);
    fwrite(frame.data(), 1, frame.size(), rec_file_);

    // Segment rotation by time
    frames_in_segment_++;
    if (frames_in_segment_ >= fps_) {
        seconds_in_segment_++;
        frames_in_segment_ = 0;
        if (seconds_in_segment_ >= segment_seconds_) {
            open_new_segment();
        }
    }
}

void MjpegHub::reader_thread_fn() {
    // Launch libcamera-vid MJPEG to stdout
    char cmd[256];
    snprintf(cmd, sizeof(cmd),
             "libcamera-vid --codec mjpeg --inline -t 0 --width %d --height %d -o -",
             width_, height_);
    FILE* pipe = popen(cmd, "r");
    if (!pipe) { perror("popen libcamera-vid"); run_ = false; return; }

    std::vector<unsigned char> buf(1 << 20); // 1MB read buffer
    std::vector<unsigned char> frame;
    bool in_frame = false;

    // Start first segment
    open_new_segment();

    while (run_) {
        size_t n = fread(buf.data(), 1, buf.size(), pipe);
        if (n == 0) break;

        for (size_t i = 0; i < n; i++) {
            unsigned char c = buf[i];

            if (!in_frame) {
                if (i + 1 < n && c == 0xFF && buf[i + 1] == 0xD8) { // SOI
                    frame.clear();
                    frame.push_back(c);
                    in_frame = true;
                }
            } else {
                frame.push_back(c);
                size_t sz = frame.size();
                if (sz >= 2 && frame[sz - 2] == 0xFF && frame[sz - 1] == 0xD9) { // EOI
                    // 1) record to disk (rotating)
                    rotate_recording_if_needed(frame);

                    // 2) fan-out to sinks
                    {
                        std::lock_guard<std::mutex> lk(sinks_m_);
                        for (auto &p : sinks_) {
                            if (p.second) p.second(frame);
                        }
                    }

                    // 3) enqueue for motion detection (bounded queue to avoid backlog)
                    {
                        std::lock_guard<std::mutex> lk(motion_mtx_);
                        // Keep the queue short; drop oldest if necessary
                        constexpr size_t MAX_Q = 5;
                        if (motion_queue_.size() >= MAX_Q) {
                            motion_queue_.pop();
                        }
                        motion_queue_.push(frame);
                    }
                    motion_cv_.notify_one();

                    in_frame = false;
                }
            }
        }
    }

    pclose(pipe);
    close_segment();
    run_ = false;

    // Wake motion thread to let it exit if we're stopping
    motion_cv_.notify_all();
}

void MjpegHub::motion_thread_fn() {
    // Motion detector
    MotionDetector detector(width_, height_, /*thresh*/25.0, /*minChange*/0.02);

    // Load YOLO model once
    cv::dnn::Net net = cv::dnn::readNetFromONNX("model.onnx");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // use GPU later if available

    while (true) {
        std::vector<unsigned char> jpeg;

        {
            std::unique_lock<std::mutex> lk(motion_mtx_);
            motion_cv_.wait(lk, [&] {
                return !motion_running_ || !motion_queue_.empty();
            });

            if (!motion_running_ && motion_queue_.empty())
                break;

            jpeg = std::move(motion_queue_.front());
            motion_queue_.pop();
        }

        // Detect motion
        if (detector.detect(jpeg)) {
            printf("Motion detected at %s\n", now_timestamp().c_str());
            fflush(stdout);

            // --- Decode JPEG to Mat
            cv::Mat raw = cv::imdecode(jpeg, cv::IMREAD_COLOR);
            if (raw.empty()) continue;

            // --- YOLO inference
            cv::Mat blob = cv::dnn::blobFromImage(raw, 1/255.0, cv::Size(640,640), cv::Scalar(), true, false);
            net.setInput(blob);
            std::vector<cv::Mat> outs;
            net.forward(outs, net.getUnconnectedOutLayersNames());

            // --- Parse results
            float* data = (float*)outs[0].data;
            int dimensions = 85; // YOLOv5: [x,y,w,h,conf,cls...]
            int rows = outs[0].size[1];
            for (int i = 0; i < rows; i++) {
                float conf = data[4];
                if (conf > 0.5) { // confidence threshold
                    float x = data[0] * raw.cols;
                    float y = data[1] * raw.rows;
                    float w = data[2] * raw.cols;
                    float h = data[3] * raw.rows;
                    int left = (int)(x - w/2);
                    int top = (int)(y - h/2);
                    cv::rectangle(raw, cv::Rect(left, top, (int)w, (int)h), cv::Scalar(0,255,0), 2);
                }
                data += dimensions;
            }

            // --- Re-encode with boxes
            std::vector<uchar> buf;
            cv::imencode(".jpg", raw, buf);

            // --- Send to sinks (with boxes)
            {
                std::lock_guard<std::mutex> lk(sinks_m_);
                for (auto &p : sinks_) {
                    if (p.second) p.second(buf);
                }
            }
        }
    }
}