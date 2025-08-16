#include "http_server.hpp"
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <iostream>

static std::string key(const std::string& m, const std::string& p){return m+" "+p;}

bool HttpServer::start(int port){
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) { perror("socket"); return false; }
    int opt=1; setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in addr{}; addr.sin_family=AF_INET; addr.sin_addr.s_addr=INADDR_ANY; addr.sin_port=htons(port);
    if (bind(server_fd_, (sockaddr*)&addr, sizeof(addr))<0){ perror("bind"); close(server_fd_); return false; }
    if (listen(server_fd_, 16)<0){ perror("listen"); close(server_fd_); return false; }
    run_ = true;
    th_ = std::thread(&HttpServer::loop, this);
    std::cout<<"HTTP server listening on port "<<port<<"\n";
    return true;
}

void HttpServer::stop(){
    run_ = false;
    if (server_fd_>=0){ close(server_fd_); server_fd_=-1; }
    if (th_.joinable()) th_.join();
}

void HttpServer::add_route(const std::string& method, const std::string& path, RouteHandler h){
    routes_[key(method,path)] = std::move(h);
}

void HttpServer::loop(){
    while (run_){
        int cfd = accept(server_fd_, nullptr, nullptr);
        if (cfd<0){ if(!run_) break; perror("accept"); continue; }
        // Handle per connection in a detached thread
        std::thread([this,cfd](){
            HttpRequest req; HttpResponse res;
            if (!read_request(cfd, req)){ close(cfd); return; }
            auto it = routes_.find(key(req.method, req.path));
            if (it != routes_.end()){
                it->second(cfd, req, res);
                if (!res.is_stream) send_response(cfd, res);
            } else {
                res.status=404; res.body="Not Found";
                send_response(cfd, res);
            }
            close(cfd);
        }).detach();
    }
}

static bool parse_request_line(const std::string& line, HttpRequest& req){
    std::istringstream iss(line);
    if(!(iss>>req.method)) return false;
    std::string target; if(!(iss>>target)) return false;
    size_t q = target.find('?');
    if (q==std::string::npos){ req.path = target; }
    else { req.path = target.substr(0,q); req.query = target.substr(q+1); }
    return true;
}

bool HttpServer::read_request(int fd, HttpRequest& req){
    // naive read until double CRLF
    std::string data; char buf[2048];
    ssize_t n;
    while ((n=recv(fd,buf,sizeof(buf),MSG_DONTWAIT))>0) data.append(buf,n);
    // if nothing yet, block once
    if (data.empty()) { n=recv(fd,buf,sizeof(buf),0); if (n<=0) return false; data.append(buf,n); }

    size_t header_end = data.find("\r\n\r\n");
    while (header_end==std::string::npos){
        n=recv(fd,buf,sizeof(buf),0); if(n<=0) break; data.append(buf,n);
        header_end = data.find("\r\n\r\n");
    }
    if (header_end==std::string::npos) return false;

    std::istringstream ss(data.substr(0, header_end));
    std::string line; if(!std::getline(ss,line)) return false;
    if (line.size() && line.back()=='\r') line.pop_back();
    if(!parse_request_line(line, req)) return false;

    while (std::getline(ss,line)){
        if (line.size() && line.back()=='\r') line.pop_back();
        size_t c=line.find(':'); if(c!=std::string::npos){
            std::string k=line.substr(0,c), v=line.substr(c+1);
            while (!v.empty() && (v.front()==' '||v.front()=='\t')) v.erase(v.begin());
            req.headers[k]=v;
        }
    }

    // Body (Content-Length)
    size_t cl=0;
    auto it=req.headers.find("Content-Length");
    if(it!=req.headers.end()) cl = std::stoul(it->second);
    size_t have = data.size() - (header_end+4);
    req.body = data.substr(header_end+4);
    while (req.body.size()<cl){
        n=recv(fd,buf,sizeof(buf),0); if(n<=0) break; req.body.append(buf,n);
    }
    return true;
}

void HttpServer::send_response(int fd, const HttpResponse& res){
    std::ostringstream hdr;
    hdr<<"HTTP/1.1 "<<res.status<<" OK\r\n";
    for (auto &kv: res.headers) hdr<<kv.first<<": "<<kv.second<<"\r\n";
    hdr<<"Content-Length: "<<res.body.size()<<"\r\n\r\n";
    auto s = hdr.str();
    send(fd, s.data(), s.size(), 0);
    if (!res.body.empty()) send(fd, res.body.data(), res.body.size(), 0);
}

