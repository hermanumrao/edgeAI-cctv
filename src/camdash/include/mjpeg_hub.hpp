#pragma once
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <string>
#include <functional>
#include <queue>
#include <thread>
#include <condition_variable>

class MjpegHub {
public:
    MjpegHub();
    ~MjpegHub();

    bool start(int width, int height, int fps);
    void stop();

    // Reader callback gets whole JPEG frames
    using FrameSink = std::function<void(const std::vector<unsigned char>&)>;
    // Register a consumer; returns an ID; call remove_sink(id) when done
    int add_sink(FrameSink cb);
    void remove_sink(int id);

private:
    std::queue<std::vector<unsigned char>> motion_queue_;
    std::mutex motion_mtx_;
    std::condition_variable motion_cv_;
    std::thread motion_thread_;
    bool motion_running_ = false;

    void motion_thread_fn();

    void reader_thread_fn();
    void rotate_recording_if_needed(const std::vector<unsigned char>& frame);

    int width_, height_, fps_;
    std::thread th_;
    std::atomic<bool> run_{false};

    // sinks
    std::mutex sinks_m_;
    int next_sink_id_ = 1;
    std::vector<std::pair<int, FrameSink>> sinks_;

    // recording
    int segment_seconds_;
    int seconds_in_segment_ = 0;
    int frames_in_segment_ = 0;
    FILE* rec_file_ = nullptr;
    std::string rec_path_;

    // helper
    void open_new_segment();
    void close_segment();
};

