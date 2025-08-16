#include <unistd.h>  // for pause()
#include "config.hpp"
#include "auth.hpp"
#include "mjpeg_hub.hpp"
#include "http_server.hpp"
#include "routes.hpp"
#include <iostream>

int main(){
    Auth auth;
    MjpegHub hub;
    if (!hub.start(cfg::CAM_WIDTH, cfg::CAM_HEIGHT, cfg::CAM_FPS)) {
        std::cerr<<"Failed to start camera hub\n"; return 1;
    }

    HttpServer srv;
    if (!srv.start(cfg::SERVER_PORT)) {
        std::cerr<<"Failed to start HTTP server\n"; return 1;
    }

    register_routes(srv, auth, hub);
    std::cout<<"Open http://<pi-ip>:"<<cfg::SERVER_PORT<<"/  (user: "<<cfg::USERNAME<<")\n";

    // Block forever
    pause();
    return 0;
}

