// recorder.cpp
#include "recorder.hpp"
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cstdint>

Recorder::Recorder(const std::string& dir_, int segment_minutes_)
    : dir(dir_), segment_minutes(segment_minutes_) {
    std::filesystem::create_directories(dir);
    open_new_file();
}

void Recorder::open_new_file() {
    if (out.is_open()) out.close();

    auto now = std::time(nullptr);
    std::ostringstream oss;
    oss << dir << "/rec_" 
        << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S")
        << ".mjpg";
    out.open(oss.str(), std::ios::binary);

    start_time = std::chrono::steady_clock::now();
}

void Recorder::push_frame(const std::vector<unsigned char>& jpeg) {
    std::lock_guard<std::mutex> lock(mtx);
    auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(
        std::chrono::steady_clock::now() - start_time
    ).count();

    if (elapsed >= segment_minutes) {
        open_new_file();
    }

    uint32_t len = jpeg.size();
    out.write(reinterpret_cast<char*>(&len), sizeof(len));
    out.write(reinterpret_cast<const char*>(jpeg.data()), len);
}

