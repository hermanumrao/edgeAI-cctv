#include "routes.hpp"
#include "config.hpp"
#include "utils.hpp"

#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h> // <-- Added
		   
// Ignore SIGPIPE globally to avoid crashes when client disconnects
struct SigPipeIgnore {
    SigPipeIgnore() { signal(SIGPIPE, SIG_IGN); }
} _sigpipe_ignore;


// --- HTML pages ---
static std::string page_login(const std::string& msg) {
    return R"(<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cam Login</title>
<style>
body { font-family: Arial, sans-serif; background:#0d1117; color:#c9d1d9; display:flex; justify-content:center; align-items:center; height:100vh; }
form { background:#161b22; padding:24px; border-radius:8px; box-shadow:0 0 12px rgba(0,0,0,0.5); }
h2 { margin-top:0; }
input { width:100%; padding:8px; margin:6px 0; border:none; border-radius:4px; background:#0d1117; color:#c9d1d9; }
button { background:#238636; color:#fff; padding:8px 12px; border:none; border-radius:4px; cursor:pointer; }
button:hover { background:#2ea043; }
.error { color:#f85149; margin-bottom:8px; }
</style>
</head>
<body>
<form method="POST" action="/login">
<h2>Login</h2>)"
+ (msg.empty() ? "" : "<div class='error'>" + msg + "</div>") +
R"(<label>User:</label>
<input name="u" autocomplete="username">
<label>Pass:</label>
<input name="p" type="password" autocomplete="current-password">
<br><br>
<button>Login</button>
</form>
</body>
</html>)";
}

static std::string page_home(const std::string& cookie) {
    return "<!doctype html><html><head><meta name=viewport content='width=device-width,initial-scale=1'>"
           "<title>CamDash</title>"
           "<style>"
           "body{font-family:Arial,sans-serif;margin:0;background:#0d1117;color:#c9d1d9}"
           ".nav{padding:12px;background:#161b22;display:flex;gap:16px;align-items:center}"
           ".nav a{color:#58a6ff;text-decoration:none} .nav b{color:#f0f6fc}"
           ".grid{display:grid;gap:16px;padding:16px;max-width:900px;margin:auto}"
           "img{max-width:100%;border-radius:8px;box-shadow:0 0 8px rgba(0,0,0,0.5)}"
           "</style></head><body>"
           "<div class=nav><b>CamDash</b><a href='/'>Live</a><a href='/videos'>Recordings</a></div>"
           "<div class=grid><h3>Live Stream</h3>"
           "<img src='/live?tok=" + http_escape(cookie) + "'/>"
           "</div></body></html>";
}

static std::string page_videos(const std::string& cookie) {
    namespace fs = std::filesystem;
    std::string html =
        "<!doctype html><html><head><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Recordings</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:0;background:#0d1117;color:#c9d1d9}"
        ".nav{padding:12px;background:#161b22;display:flex;gap:16px;align-items:center}"
        ".nav a{color:#58a6ff;text-decoration:none} .nav b{color:#f0f6fc}"
        "table{width:100%;border-collapse:collapse;margin-top:16px}"
        "th,td{padding:10px;border-bottom:1px solid #30363d}"
        "tr:hover{background:#161b22}"
        "a.play{color:#58a6ff}"
        "</style>"
        "<script>"
        "function openPlayer(file,tok){"
        "  const win=window.open('', '_blank', 'width=800,height=600');"
        "  win.document.write(`<!doctype html><html><head><title>Playback</title>"
        "  <style>body{margin:0;background:#000;color:#fff;text-align:center}"
        "  video{width:100%;height:auto;max-height:90vh}"
        "  .close-btn{position:fixed;top:10px;right:10px;background:#f33;color:#fff;border:none;padding:8px;cursor:pointer}</style>"
        "  </head><body>"
        "  <button class='close-btn' onclick='window.close()'>Close</button>"
        "  <img src='/play?f=${file}&tok=${tok}'/>"
        "  </body></html>`);"
        "}"
        "</script></head><body>"
        "<div class=nav><b>CamDash</b><a href='/'>Live</a><a href='/videos'>Recordings</a></div>"
        "<div style='padding:16px;max-width:900px;margin:auto'><h3>Recordings</h3>"
        "<table><tr><th>File</th><th>Size</th><th>Play</th></tr>";

    for (auto &e : fs::directory_iterator(cfg::RECORD_DIR)) {
        if (!e.is_regular_file()) continue;
        auto p = e.path();
        if (p.extension() != ".mjpg") continue;
        auto sz = fs::file_size(p);
        html += "<tr><td>" + p.filename().string() + "</td>"
                "<td>" + std::to_string(sz / 1024) + " KB</td>"
                "<td><a class='play' href='#' onclick=\"openPlayer('" + http_escape(p.filename().string()) +
                "','" + http_escape(cookie) + "')\">Play</a></td></tr>";
    }
    html += "</table></div></body></html>";
    return html;
}

static std::string page_player(const std::string& file, const std::string& tok) {
    return "<!doctype html><html><head>"
           "<meta name='viewport' content='width=device-width,initial-scale=1'>"
           "<title>Playing: " + file + "</title>"
           "<style>"
           "body{margin:0;background:#0d1117;display:flex;flex-direction:column;height:100vh}"
           ".nav{background:#161b22;padding:12px;display:flex;align-items:center;gap:12px}"
           ".nav button{background:#f85149;border:none;color:white;padding:6px 12px;"
           "border-radius:4px;cursor:pointer}"
           ".nav button:hover{background:#da3633}"
           ".video-container{flex:1;display:flex;justify-content:center;align-items:center}"
           "img{max-width:100%;max-height:100%;background:black}"
           "</style></head><body>"
           "<div class='nav'><button onclick=\"window.location.href='/videos?tok=" + tok + "'\">Close</button>"
           "<div style='color:#c9d1d9'>Playing: " + file + "</div></div>"
           "<div class='video-container'><img src='/streamfile?f=" + file + "&tok=" + tok + "'></div>"
           "</body></html>";
}

// --- helpers ---
static void parse_form(const std::string& body, std::string& u, std::string& p) {
    auto up = body.find("u=");
    auto pp = body.find("p=");
    if (up != std::string::npos) {
        auto amp = body.find('&', up);
        u = body.substr(up + 2, (amp == std::string::npos ? body.size() : amp) - (up + 2));
    }
    if (pp != std::string::npos) {
        auto amp = body.find('&', pp);
        p = body.substr(pp + 2, (amp == std::string::npos ? body.size() : amp) - (pp + 2));
    }
}

static std::string cookie_from_headers(const HttpRequest& req) {
    auto it = req.headers.find("Cookie");
    if (it == req.headers.end()) return "";
    auto &v = it->second;
    auto pos = v.find(cfg::COOKIE_NAME + "=");
    if (pos == std::string::npos) return "";
    pos += cfg::COOKIE_NAME.size() + 1;
    auto end = v.find(';', pos);
    return v.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
}

// --- register routes ---
void register_routes(HttpServer& srv, Auth& auth, MjpegHub& hub) {
    srv.add_route("GET","/",[&](int fd, const HttpRequest& req, HttpResponse& res){
        auto cookie = cookie_from_headers(req);
        if (!auth.check_cookie(cookie)) {
            res.headers["Content-Type"]="text/html";
            res.body = page_login("");
            return;
        }
        res.headers["Content-Type"]="text/html";
        res.body = page_home(cookie);
    });

    srv.add_route("POST","/login",[&](int fd, const HttpRequest& req, HttpResponse& res){
        std::string u,p; parse_form(req.body,u,p);
        if (auth.check_credentials(u,p)) {
            auto tok = auth.new_session();
            res.headers["Set-Cookie"] = cfg::COOKIE_NAME+"="+tok+"; HttpOnly; SameSite=Lax";
            res.headers["Location"] = "/";
            res.status = 302;
        } else {
            res.headers["Content-Type"]="text/html";
            res.body = page_login("Invalid credentials");
        }
    });

    srv.add_route("GET","/videos",[&](int fd, const HttpRequest& req, HttpResponse& res){
        auto cookie = cookie_from_headers(req);
        if (!auth.check_cookie(cookie)) {
            res.headers["Content-Type"]="text/html";
            res.body = page_login("");
            return;
        }
        res.headers["Content-Type"]="text/html";
        res.body = page_videos(cookie);
    });


    srv.add_route("GET","/live",[&](int fd, const HttpRequest& req, HttpResponse& res){
        std::string tok;
        auto qpos = req.query.find("tok=");
        if (qpos!=std::string::npos) tok = req.query.substr(qpos+4);
        if (!auth.check_cookie(tok)) {
            res.status=403; res.body="Forbidden"; return;
        }
        res.is_stream = true;
        std::string hdr = "HTTP/1.1 200 OK\r\n"
                          "Cache-Control: no-cache\r\nPragma: no-cache\r\n"
                          "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        if (send(fd, hdr.c_str(), hdr.size(), MSG_NOSIGNAL) <= 0) return;

        int sink_id = hub.add_sink([fd](const std::vector<unsigned char>& jpeg){
            char head[128];
            int n = snprintf(head, sizeof(head),
                             "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %zu\r\n\r\n",
                             jpeg.size());
            if (send(fd, head, n, MSG_NOSIGNAL) <= 0) return;
            if (send(fd, (const char*)jpeg.data(), jpeg.size(), MSG_NOSIGNAL) <= 0) return;
            static const char crlf[2] = {'\r','\n'};
            send(fd, crlf, 2, MSG_NOSIGNAL);
        });

        char tmp;
        while (recv(fd, &tmp, 1, MSG_PEEK) > 0) { usleep(200000); }
        hub.remove_sink(sink_id);
    });

    // Play recording route - fixed send() calls
    srv.add_route("GET","/play",[&](int fd, const HttpRequest& req, HttpResponse& res){
        std::string tok; auto tpos = req.query.find("tok=");
        if (tpos!=std::string::npos) tok = req.query.substr(tpos+4);
        if (!auth.check_cookie(tok)) { res.status=403; res.body="Forbidden"; return; }

        std::string fn; auto fpos = req.query.find("f=");
        if (fpos!=std::string::npos){
            fn = req.query.substr(fpos+2);
            auto amp = fn.find('&');
            if (amp!=std::string::npos) fn = fn.substr(0, amp);
        }
        if (fn.find("..")!=std::string::npos){ res.status=400; res.body="Bad filename"; return; }

        std::string path = cfg::RECORD_DIR + "/" + fn;
        std::ifstream in(path, std::ios::binary);
        if (!in){ res.status=404; res.body="Not found"; return; }

        res.is_stream = true;
        std::string hdr = "HTTP/1.1 200 OK\r\n"
                          "Cache-Control: no-cache\r\nPragma: no-cache\r\n"
                          "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        if (send(fd, hdr.c_str(), hdr.size(), MSG_NOSIGNAL) <= 0) return;

        while (true){
            uint32_t len=0;
            in.read((char*)&len, sizeof(len));
            if (!in || len==0) break;
            std::string jpeg(len, '\0');
            in.read(jpeg.data(), len);
            if (!in) break;
            char head[128];
            int n = snprintf(head, sizeof(head),
                             "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                             len);
            if (send(fd, head, n, MSG_NOSIGNAL) <= 0) break;
            if (send(fd, jpeg.data(), jpeg.size(), MSG_NOSIGNAL) <= 0) break;
            static const char crlf[2] = {'\r','\n'};
            if (send(fd, crlf, 2, MSG_NOSIGNAL) <= 0) break;
            usleep(1000000 / cfg::CAM_FPS);
        }
    });


    // New: Raw MJPEG stream for recordings
    srv.add_route("GET","/streamfile",[&](int fd, const HttpRequest& req, HttpResponse& res){
        std::string tok; auto tpos = req.query.find("tok=");
        if (tpos!=std::string::npos) tok = req.query.substr(tpos+4);
        if (!auth.check_cookie(tok)) { res.status=403; res.body="Forbidden"; return; }

        std::string fn; auto fpos = req.query.find("f=");
        if (fpos!=std::string::npos){
            fn = req.query.substr(fpos+2);
            auto amp = fn.find('&');
            if (amp!=std::string::npos) fn = fn.substr(0, amp);
        }
        if (fn.find("..")!=std::string::npos){ res.status=400; res.body="Bad filename"; return; }

        std::string path = cfg::RECORD_DIR + "/" + fn;
        std::ifstream in(path, std::ios::binary);
        if (!in){ res.status=404; res.body="Not found"; return; }

        res.is_stream = true;
        std::string hdr = "HTTP/1.1 200 OK\r\n"
                          "Cache-Control: no-cache\r\nPragma: no-cache\r\n"
                          "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        send(fd, hdr.c_str(), hdr.size(), 0);

        while (true){
            uint32_t len=0;
            in.read((char*)&len, sizeof(len));
            if (!in || len==0) break;
            std::string jpeg(len, '\0');
            in.read(jpeg.data(), len);
            if (!in) break;
            char head[128];
            int n = snprintf(head, sizeof(head),
                             "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                             len);
            if (send(fd, head, n, 0) <= 0) break;
            if (send(fd, jpeg.data(), jpeg.size(), 0) <= 0) break;
            static const char crlf[2] = {'\r','\n'};
            if (send(fd, crlf, 2, 0) <= 0) break;
            usleep(1000000 / cfg::CAM_FPS);
        }
    });
}

