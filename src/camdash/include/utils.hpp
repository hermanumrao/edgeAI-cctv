#pragma once
#include <string>
#include <ctime>

std::string now_timestamp();            // e.g. "2025-08-16 14:32:10"
std::string now_timestamp_filename();         // e.g. 2025-08-15_12-34-56
bool ensure_dir(const std::string& path);
std::string http_escape(const std::string& s);

