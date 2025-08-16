#include "utils.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iomanip>
#include <sstream>

std::string now_timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm); // thread-safe
    char buf[64];
    if (std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm) == 0)
        return std::string();
    return std::string(buf);
}

std::string now_timestamp_filename() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

bool ensure_dir(const std::string& path) {
    struct stat st{};
    if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) return true;
    return mkdir(path.c_str(), 0755) == 0;
}

static bool is_unreserved(char c) {
    return (isalnum((unsigned char)c) || c=='-'||c=='_'||c=='.'||c=='~');
}

std::string http_escape(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        if (is_unreserved(c)) o<<c;
        else { o<<'%'<<std::uppercase<<std::hex<<std::setw(2)<<std::setfill('0')<<(int)c<<std::nouppercase<<std::dec; }
    }
    return o.str();
}

