#pragma once
#include "http_server.hpp"
#include "auth.hpp"
#include "mjpeg_hub.hpp"

void register_routes(HttpServer& srv, Auth& auth, MjpegHub& hub);

