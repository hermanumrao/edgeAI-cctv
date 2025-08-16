#pragma once
#include <functional>
#include <string>
#include <unordered_map>
#include <thread>
#include <atomic>

struct HttpRequest {
    std::string method;
    std::string path;
    std::string query;
    std::unordered_map<std::string,std::string> headers;
    std::string body;
};

struct HttpResponse {
    int status = 200;
    std::unordered_map<std::string,std::string> headers;
    std::string body;
    bool is_stream = false; // if true, handler writes directly to socket
};

using RouteHandler = std::function<void(int client_fd, const HttpRequest&, HttpResponse&)>;

class HttpServer {
public:
    bool start(int port);
    void stop();
    void add_route(const std::string& method, const std::string& path, RouteHandler h);

private:
    int server_fd_ = -1;
    std::thread th_;
    std::atomic<bool> run_{false};
    std::unordered_map<std::string, RouteHandler> routes_;

    void loop();
    static bool read_request(int fd, HttpRequest& req);
    static void send_response(int fd, const HttpResponse& res);
};

