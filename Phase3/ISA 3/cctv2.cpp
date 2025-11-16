#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp> // HOG
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <regex>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <map>
#include <algorithm> // for std::replace

namespace fs = std::filesystem;
using namespace std::chrono;

// Config
static int HTTP_PORT = 8080;
static int CAM_INDEX = 0;
static int FRAME_WIDTH = 640;
static int FRAME_HEIGHT = 480;
static int MJPEG_JPG_QUALITY = 80;
static int TARGET_FPS = 20;
static int RECORD_FPS = 20;
static int RECORD_SECONDS = 10;
static std::string RECORD_DIR = "recordings";
static std::string EVENTS_CSV = "recordings/events.csv";
static bool RECORD_ENABLED = true;

static double BG_LEARN_RATE = 0.02;
static int THRESH_DELTA = 25;
static int MIN_AREA = 1200;
static int DILATE_ITERS = 2;
static int CONFIRM_FRAMES = 4;
static int END_FRAMES = 10;
static double MIN_AR = 0.2;
static double MAX_AR = 1.2;

// HOG tuning
static int HOG_INTERVAL = 5; // run HOG every N frames

std::atomic<bool> g_running{true};
std::mutex g_jpeg_mtx;
std::vector<uchar> g_latest_jpeg;
std::mutex g_frame_mtx;
cv::Mat g_latest_frame_bgr;

std::mutex g_rec_mtx;
cv::VideoWriter g_writer;
std::string g_current_filename;
int g_frames_in_segment = 0;

struct Event { double start_sec; double end_sec; bool open; };
std::mutex g_evt_mtx;
std::vector<Event> g_segment_events;

// Pending CSV jobs (filename -> events)
std::mutex g_pending_csv_mtx;
std::vector<std::pair<std::string, std::vector<Event>>> g_pending_csv;

// HOG detector
cv::HOGDescriptor g_hog;

std::string timestamp_now_str() {
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

void ensure_dir(const std::string& d) {
    if (!fs::exists(d)) fs::create_directories(d);
}

void append_events_csv_internal(const std::string& filename, const std::vector<Event>& evts) {
    // internal helper: directly append to CSV (called from csv thread only)
    if (evts.empty()) return;
    std::ofstream f(EVENTS_CSV, std::ios::app);
    if (!f.good()) return;
    std::string base = fs::path(filename).filename().string();
    for (auto &e : evts) {
        double endv = e.open ? e.start_sec : e.end_sec;
        f << base << "," << std::fixed << std::setprecision(2)
          << e.start_sec << "," << endv << "\n";
    }
}

void csv_writer_thread() {
    while (g_running.load()) {
        std::pair<std::string, std::vector<Event>> job;
        bool have = false;
        {
            std::lock_guard<std::mutex> lk(g_pending_csv_mtx);
            if (!g_pending_csv.empty()) {
                job = std::move(g_pending_csv.back());
                g_pending_csv.pop_back();
                have = true;
            }
        }
        if (have) {
            append_events_csv_internal(job.first, job.second);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    // Final flush on exit
    {
        std::lock_guard<std::mutex> lk(g_pending_csv_mtx);
        while (!g_pending_csv.empty()) {
            auto jb = std::move(g_pending_csv.back());
            g_pending_csv.pop_back();
            append_events_csv_internal(jb.first, jb.second);
        }
    }
}

// rotate_segment: open new writer off the critical section and swap quickly
void rotate_segment_nonblocking(int width, int height) {
    // Prepare new writer first (may take time)
    std::string new_filename = RECORD_DIR + "/rec_" + timestamp_now_str() + ".mp4";
    int fourcc = cv::VideoWriter::fourcc('a','v','c','1');
    cv::VideoWriter new_writer;
    new_writer.open(new_filename, fourcc, RECORD_FPS, cv::Size(width, height));
    if (!new_writer.isOpened()) {
        fourcc = cv::VideoWriter::fourcc('M','J','P','G');
        new_writer.open(new_filename, fourcc, RECORD_FPS, cv::Size(width, height));
    }

    // Swap quickly with the global under lock
    std::string old_filename;
    {
        std::lock_guard<std::mutex> lk(g_rec_mtx);
        if (g_writer.isOpened()) {
            g_writer.release();
        }
        g_writer = std::move(new_writer);
        old_filename = g_current_filename;
        g_current_filename = new_filename;
        g_frames_in_segment = 0;
    }

    // Move events to pending CSV queue (protected)
    {
        std::lock_guard<std::mutex> ek(g_evt_mtx);
        if (!g_segment_events.empty()) {
            std::lock_guard<std::mutex> pk(g_pending_csv_mtx);
            g_pending_csv.push_back({ old_filename.empty() ? g_current_filename : old_filename, g_segment_events });
            g_segment_events.clear();
        }
    }
}

// overload - to call at startup when no writer yet: open writer directly
void open_initial_writer(int width, int height) {
    std::string new_filename = RECORD_DIR + "/rec_" + timestamp_now_str() + ".mp4";
    int fourcc = cv::VideoWriter::fourcc('a','v','c','1');
    g_writer.open(new_filename, fourcc, RECORD_FPS, cv::Size(width, height));
    if (!g_writer.isOpened()) {
        fourcc = cv::VideoWriter::fourcc('M','J','P','G');
        g_writer.open(new_filename, fourcc, RECORD_FPS, cv::Size(width, height));
    }
    g_current_filename = new_filename;
    g_frames_in_segment = 0;
}

void update_events(bool detected, double segment_time_sec) {
    std::lock_guard<std::mutex> lk(g_evt_mtx);
    static int consec_on = 0, consec_off = 0;
    static bool in_event = false;
    static double event_start = 0.0;

    if (detected) { consec_on++; consec_off = 0; }
    else { consec_off++; consec_on = 0; }

    if (!in_event && consec_on >= CONFIRM_FRAMES) {
        in_event = true;
        event_start = segment_time_sec;
        g_segment_events.push_back({event_start, event_start, true});
    }

    if (in_event && consec_off >= END_FRAMES) {
        in_event = false;
        for (auto &e : g_segment_events) {
            if (e.open) {
                e.open = false;
                e.end_sec = segment_time_sec;
                break;
            }
        }
    }
}

bool encode_jpeg(const cv::Mat& bgr, std::vector<uchar>& out, int q=80) {
    std::vector<int> p = {cv::IMWRITE_JPEG_QUALITY, q};
    return cv::imencode(".jpg", bgr, out, p);
}

// Capture thread
void capture_thread() {
    // Build GStreamer pipeline using FRAME_WIDTH/HEIGHT and TARGET_FPS
    std::ostringstream gst;
    gst << "libcamerasrc ! video/x-raw,width=" << FRAME_WIDTH
        << ",height=" << FRAME_HEIGHT
        << ",framerate=" << TARGET_FPS << "/1 ! "
        << "videoconvert ! video/x-raw,format=BGR ! appsink";

    cv::VideoCapture cap(gst.str(), cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera\n";
        g_running = false;
        return;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    ensure_dir(RECORD_DIR);

    if (RECORD_ENABLED) {
        // Open initial writer
        open_initial_writer(width, height);
    }

    cv::Mat frame, gray, bg, bg8, diff, thresh;
    bool bg_init=false;
    auto next_tick = steady_clock::now();

    int hog_frame_counter = 0;

    while (g_running.load()) {
        if (!cap.read(frame) || frame.empty()) {
            // short sleep on read failure to avoid busy loop
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        {
            std::lock_guard<std::mutex> lk(g_frame_mtx);
            g_latest_frame_bgr = frame.clone();
        }

        // Preprocess
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5,5),0);

        if (!bg_init) { gray.convertTo(bg, CV_32F); bg_init=true; }
        else cv::accumulateWeighted(gray,bg,BG_LEARN_RATE);
        bg.convertTo(bg8,CV_8U);
        cv::absdiff(gray,bg8,diff);
        cv::threshold(diff,thresh,THRESH_DELTA,255,cv::THRESH_BINARY);
        cv::dilate(thresh,thresh,cv::Mat(),cv::Point(-1,-1),DILATE_ITERS);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

        bool humanish_motion = false;

        // Motion-based shape detector (draw rectangles)
        for (auto &c: contours) {
            double area=cv::contourArea(c);
            if (area<MIN_AREA) continue;
            cv::Rect r=cv::boundingRect(c);
            double ar=(double)r.width/std::max(1,r.height);
            if (ar>=MIN_AR && ar<=MAX_AR) {
                cv::rectangle(frame,r,cv::Scalar(0,255,0),2);
                humanish_motion = true;
            }
        }

        // HOG detection every HOG_INTERVAL frames
        bool humanish_hog = false;
        hog_frame_counter++;
        if (hog_frame_counter >= HOG_INTERVAL) {
            hog_frame_counter = 0;
            std::vector<cv::Rect> found;
            g_hog.detectMultiScale(gray, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);
            if (!found.empty()) {
                humanish_hog = true;
                for (auto &p : found) {
                    cv::rectangle(frame, p, cv::Scalar(0,0,255), 2);
                }
                std::cout << "[HOG] Human detected at " << timestamp_now_str() << std::endl;
            }
        }

        // Combine detectors
        bool humanish = humanish_motion || humanish_hog;

        // RECORDING
        if (RECORD_ENABLED) {
            bool wrote = false;
            {
                std::lock_guard<std::mutex> lk(g_rec_mtx);
                if (g_writer.isOpened()) {
                    g_writer.write(frame);
                    g_frames_in_segment++;
                    wrote = true;
                }
            }
            if (wrote) {
                double seg_time = (double)g_frames_in_segment / std::max(1, RECORD_FPS);
                update_events(humanish, seg_time);
                if (seg_time >= RECORD_SECONDS) {
                    rotate_segment_nonblocking(width, height);
                }
            }
        }

        // JPEG encode for mjpeg stream
        std::vector<uchar> jpg;
        encode_jpeg(frame, jpg, MJPEG_JPG_QUALITY);
        {
            std::lock_guard<std::mutex> lk(g_jpeg_mtx);
            g_latest_jpeg.swap(jpg);
        }

        // maintain target fps
        next_tick += milliseconds(1000 / std::max(1, TARGET_FPS));
        std::this_thread::sleep_until(next_tick);
    }

    // On exit: close and push remaining events
    if (RECORD_ENABLED) {
        {
            std::lock_guard<std::mutex> lk(g_rec_mtx);
            if (g_writer.isOpened()) {
                g_writer.release();
            }
        }
        {
            std::lock_guard<std::mutex> ek(g_evt_mtx);
            std::lock_guard<std::mutex> pk(g_pending_csv_mtx);
            if (!g_segment_events.empty()) {
                g_pending_csv.push_back({ g_current_filename, g_segment_events });
                g_segment_events.clear();
            }
        }
    }
    cap.release();
}

// --------- HTTP / HTML ---------
std::string html_index() {
    std::ostringstream o;
    o << R"(<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CCTV</title>
<style>
body{font-family:system-ui,Arial;margin:20px;background:#0b0e12;color:#e6e6e6}
.card{background:#131821;border:1px solid #1f2633;border-radius:14px;padding:16px;margin-bottom:16px;box-shadow:0 6px 20px rgba(0,0,0,0.25)}
.btn{display:inline-block;padding:8px 12px;border-radius:10px;background:#1f2937;color:#e6e6e6;text-decoration:none;border:1px solid #324055;margin:4px}
input,button{padding:6px;margin:4px;border-radius:8px;border:none;background:#1f2633;color:#e6e6e6}
</style>
</head>
<body>
<div class="card">
  <h1>CCTV Live</h1>
  <img src="/mjpeg" alt="Live stream" style="width:100%;max-width:640px;">
</div>

<div class="card">
  <h2>Network Cameras</h2>
  <form action="/add_camera" method="POST">
    <input name="name" placeholder="Camera name " required>
    <button type="submit">Add</button>
  </form>
)";

    // load cameras.txt
    ensure_dir(RECORD_DIR);
    std::ifstream camfile(RECORD_DIR + "/cameras.txt");
    std::string line;
    while (std::getline(camfile, line)) {
        auto comma = line.find(',');
        if (comma == std::string::npos) continue;
        auto name = line.substr(0, comma);
        auto url  = line.substr(comma + 1);
        o << "<a class='btn' href='" << url << "' target='_blank'>" << name << "</a>";
    }

    o << R"(
</div>
<div class="card">
  <h2>Recordings</h2>
  <a class="btn" href="/recordings">Open recordings page</a>
</div>
</body></html>)";
    return o.str();
}

std::string html_recordings() {
    // Read directory + events CSV to annotate
    struct Mark { std::string file; double s; double e; };
    std::vector<Mark> marks;
    {
        std::ifstream f(EVENTS_CSV);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string file, s, e;
            if (!std::getline(ss, file, ',')) continue;
            if (!std::getline(ss, s, ',')) continue;
            if (!std::getline(ss, e, ',')) continue;
            try {
                marks.push_back({file, std::stod(s), std::stod(e)});
            } catch (...) {}
        }
    }
    // Group marks by file
    std::map<std::string, std::vector<std::pair<double,double>>> byfile;
    for (auto &m : marks) byfile[m.file].push_back({m.s, m.e});

    std::ostringstream o;
    o <<
R"(<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Recordings</title>
<style>
body{font-family:system-ui,Arial;margin:20px;background:#0b0e12;color:#e6e6e6}
.card{background:#131821;border:1px solid #1f2633;border-radius:14px;padding:16px;margin-bottom:16px;box-shadow:0 6px 20px rgba(0,0,0,0.25)}
h1,h2{margin:8px 0}
a{color:#7db4ff}
.btn{display:inline-block;padding:8px 12px;border-radius:10px;background:#1f2937;color:#e6e6e6;text-decoration:none;border:1px solid #324055;margin-right:8px}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;background:#7a1f1f;margin:4px 6px 0 0; font-size:12px}
.row{display:flex;gap:16px;align-items:center;flex-wrap:wrap}
video{max-width:100%;width:480px;border-radius:10px;border:1px solid #1f2633}
.small{font-size:12px;color:#9aa7bd}
</style>
</head>
<body>
<div class="card">
  <h1>Recordings</h1>
  <a class="btn" href="/">← Back to Live</a>
</div>
)";

    // List files newest first
    std::vector<fs::directory_entry> files;
    if (fs::exists(RECORD_DIR)) {
        for (auto &p : fs::directory_iterator(RECORD_DIR)) {
            if (!p.is_regular_file()) continue;
            auto ext = p.path().extension().string();
            if (ext == ".mp4" || ext == ".avi" || ext == ".mkv") files.push_back(p);
        }
    }
    std::sort(files.begin(), files.end(), [](auto&a, auto&b){
        return fs::last_write_time(a) > fs::last_write_time(b);
    });

    for (auto &p : files) {
        std::string name = p.path().filename().string();
        std::string url  = "/video/" + name;
        o << "<div class='card'>\n";
        o << "  <div class='row'><h2>" << name << "</h2></div>\n";
        o << "  <div class='row'><video controls src='" << url << "'></video></div>\n";
        // Show detection tags
        auto it = byfile.find(name);
        if (it == byfile.end()) it = byfile.find(RECORD_DIR + "/" + name);
        if (it != byfile.end()) {
            o << "  <div class='row small'><div>Detected times:</div>";
            for (auto &pr : it->second) {
                o << "<span class='tag'>" << std::fixed << std::setprecision(1)
                  << pr.first << "–" << pr.second << "s</span>";
            }
            o << "</div>\n";
            o << "  <div class='small'>Tip: use the player's timeline to seek to a highlighted time.</div>\n";
        } else {
            o << "  <div class='small'>No detections recorded for this file.</div>\n";
        }
        o << "</div>\n";
    }

    o << "</body></html>";
    return o.str();
}

// Send a whole string as HTTP 200 text/html
void send_http_str(int client, const std::string& body, const std::string& mime="text/html; charset=utf-8") {
    std::ostringstream h;
    h << "HTTP/1.0 200 OK\r\n";
    h << "Content-Type: " << mime << "\r\n";
    h << "Content-Length: " << body.size() << "\r\n";
    h << "Connection: close\r\n\r\n";
    std::string head = h.str();
    send(client, head.data(), head.size(), 0);
    send(client, body.data(), body.size(), 0);
}

// Very simple static file sender (no Range support)
void send_file_simple(int client, const std::string& path, const std::string& mime) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        std::string notfound = "HTTP/1.0 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        send(client, notfound.data(), notfound.size(), 0);
        return;
    }
    f.seekg(0, std::ios::end);
    size_t len = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    std::ostringstream h;
    h << "HTTP/1.0 200 OK\r\n";
    h << "Content-Type: " << mime << "\r\n";
    h << "Content-Length: " << len << "\r\n";
    h << "Connection: close\r\n\r\n";
    std::string head = h.str();
    send(client, head.data(), head.size(), 0);

    const size_t BUFSZ = 64 * 1024;
    char buf[BUFSZ];
    while (f.good()) {
        f.read(buf, BUFSZ);
        std::streamsize got = f.gcount();
        if (got > 0) send(client, buf, (size_t)got, 0);
    }
}

std::string detect_mime(const std::string& path) {
    auto ext = fs::path(path).extension().string();
    if (ext == ".mp4") return "video/mp4";
    if (ext == ".avi") return "video/x-msvideo";
    if (ext == ".mkv") return "video/x-matroska";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".png") return "image/png";
    if (ext == ".html") return "text/html; charset=utf-8";
    return "application/octet-stream";
}

void handle_mjpeg(int client) {
    std::ostringstream h;
    h << "HTTP/1.0 200 OK\r\n";
    h << "Cache-Control: no-cache\r\nPragma: no-cache\r\n";
    h << "Connection: close\r\n";
    h << "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    std::string head = h.str();
    send(client, head.data(), head.size(), 0);

    // Send frames until disconnect
    while (g_running.load()) {
        std::vector<uchar> jpg;
        {
            std::lock_guard<std::mutex> lk(g_jpeg_mtx);
            if (!g_latest_jpeg.empty()) jpg = g_latest_jpeg;
        }
        if (!jpg.empty()) {
            std::ostringstream part;
            part << "--frame\r\n"
                 << "Content-Type: image/jpeg\r\n"
                 << "Content-Length: " << jpg.size() << "\r\n\r\n";
            std::string ph = part.str();
            if (send(client, ph.data(), ph.size(), 0) < 0) break;
            if (send(client, (const char*)jpg.data(), jpg.size(), 0) < 0) break;
            if (send(client, "\r\n", 2, 0) < 0) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / std::max(1, TARGET_FPS)));
    }
}

void client_thread(int client) {
    // Read a tiny HTTP request
    char buf[4096];
    int n = recv(client, buf, sizeof(buf)-1, 0);
    if (n <= 0) { close(client); return; }
    buf[n] = 0;
    std::string req(buf);

    // parse "GET /path HTTP/1.1"
    std::smatch m;
    std::regex r("^([A-Z]+) ([^ ]+) HTTP");
    std::string method, path;
    if (std::regex_search(req, m, r)) {
        method = m[1];
        path   = m[2];
    }

    // Handle POST /add_camera
    if (method == "POST" && path == "/add_camera") {
        auto pos = req.find("\r\n\r\n");
        std::string body;
        if (pos != std::string::npos) body = req.substr(pos + 4);

        std::smatch m2;
        std::regex r2("name=([^&]*)");
        if (std::regex_search(body, m2, r2)) {
            std::string name = m2[1];
            // basic decode: + -> space
            std::replace(name.begin(), name.end(), '+', ' ');
            // you could also decode %XX here if needed

            ensure_dir(RECORD_DIR);
            std::string url = "http://" + name + ".local:8080/";
            std::ofstream f(RECORD_DIR + "/cameras.txt", std::ios::app);
            if (f.good()) {
                f << name << "," << url << "\n";
            }
        }

        send_http_str(client,
            "<html><body><h2>Camera added</h2>"
            "<a href='/'>Back</a></body></html>");
        close(client);
        return;
    }

    // For everything else, only allow GET
    if (method != "GET") {
        std::string resp = "HTTP/1.0 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        send(client, resp.data(), resp.size(), 0);
        close(client);
        return;
    }

    if (path == "/") {
        send_http_str(client, html_index());
    } else if (path == "/recordings") {
        send_http_str(client, html_recordings());
    } else if (path == "/mjpeg") {
        handle_mjpeg(client);
    } else if (path.rfind("/video/", 0) == 0) {
        std::string fname = path.substr(std::string("/video/").size());
        std::string realp = RECORD_DIR + "/" + fname;
        send_file_simple(client, realp, detect_mime(realp));
    } else {
        std::string nf = "HTTP/1.0 404 Not Found\r\nContent-Length:0\r\nConnection: close\r\n\r\n";
        send(client, nf.data(), nf.size(), 0);
    }

    close(client);
}

void http_server_thread() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { std::perror("socket"); g_running=false; return; }

    int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(HTTP_PORT);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) { std::perror("bind"); close(server_fd); g_running=false; return; }
    if (listen(server_fd, 8) < 0) { std::perror("listen"); close(server_fd); g_running=false; return; }

    std::cerr << "[HTTP] Listening on port " << HTTP_PORT << std::endl;

    while (g_running.load()) {
        sockaddr_in cli{};
        socklen_t cl = sizeof(cli);
        int client = accept(server_fd, (sockaddr*)&cli, &cl);
        if (client < 0) {
            if (!g_running.load()) break;
            continue;
        }
        std::thread(client_thread, client).detach();
    }
    close(server_fd);
}

void on_sigint(int) {
    g_running = false;
}

int main(int argc, char** argv) {
    if (argc > 1) HTTP_PORT = std::stoi(argv[1]);

    // HOG init
    g_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    std::signal(SIGINT, on_sigint);
    std::signal(SIGTERM, on_sigint);
    signal(SIGPIPE, SIG_IGN);

    // Start threads
    std::thread csvthr(csv_writer_thread);
    std::thread capthr(capture_thread);
    std::thread httpthr(http_server_thread);

    // Join capture thread (it will run until signal)
    capthr.join();

    // Ask other threads to stop and join them
    g_running = false;
    if (httpthr.joinable()) httpthr.detach(); // accept may block; process exiting
    if (csvthr.joinable()) csvthr.join();

    return 0;
}

