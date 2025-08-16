#pragma once
#include <string>
#include <unordered_set>
#include <mutex>

class Auth {
public:
    bool check_credentials(const std::string& user, const std::string& pass) const;
    std::string new_session();
    bool check_cookie(const std::string& cookie) const;
private:
    std::unordered_set<std::string> sessions_;
    mutable std::mutex m_;
};

