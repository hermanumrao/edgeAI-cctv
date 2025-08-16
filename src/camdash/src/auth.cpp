#include "auth.hpp"
#include "config.hpp"
#include <random>

bool Auth::check_credentials(const std::string& user, const std::string& pass) const {
    return (user == cfg::USERNAME && pass == cfg::PASSWORD);
}

std::string Auth::new_session() {
    std::random_device rd; std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> dist;
    unsigned long long a = dist(gen), b = dist(gen);
    char buf[33];
    snprintf(buf, sizeof(buf), "%016llx%016llx",
             (unsigned long long)a, (unsigned long long)b);
    std::string tok(buf);
    std::lock_guard<std::mutex> lk(m_);
    sessions_.insert(tok);
    return tok;
}

bool Auth::check_cookie(const std::string& cookie) const {
    std::lock_guard<std::mutex> lk(m_);
    return sessions_.count(cookie)>0;
}

