#pragma once
#include <string>

namespace cfg {
// Network
inline constexpr int SERVER_PORT = 8080;

// Camera
inline constexpr int CAM_WIDTH  = 640;
inline constexpr int CAM_HEIGHT = 480;
inline constexpr int CAM_FPS    = 20;   // live playback pacing

// Recording
inline const std::string RECORD_DIR = "./recordings";
inline constexpr int SEGMENT_MINUTES = 1; // new file every N minutes

// Auth (demo only; change these!)
inline const std::string USERNAME = "admin";
inline const std::string PASSWORD = "admin"; // plaintext for simplicity
inline const std::string COOKIE_NAME = "camdash_sess";
}

