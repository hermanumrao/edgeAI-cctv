// recorder.hpp
#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <mutex>
#include <chrono>

class Recorder {
public:
    Recorder(const std::string& dir, int segment_minutes);
    void push_frame(const std::vector<unsigned char>& jpeg);
private:
    std::string dir;
    int segment_minutes;
    std::ofstream out;
    std::mutex mtx;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    void open_new_file();
};

