#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <csignal>

#define PORT 8080
#define USERNAME "admin"
#define PASSWORD "password"
#define MOTION_THRESHOLD 10000  // Adjust sensitivity of motion detection

using namespace cv;
using namespace std;

string loginPage =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/html\r\n\r\n"
    "<!DOCTYPE html>"
    "<html lang='en'>"
    "<head>"
    "<meta charset='UTF-8'>"
    "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
    "<title>Login</title>"
    "<style>"
    "body {"
    "    display: flex;"
    "    justify-content: center;"
    "    align-items: center;"
    "    height: 100vh;"
    "    background: linear-gradient(to right, #6a11cb, #2575fc);"
    "    font-family: Arial, sans-serif;"
    "    color: white;"
    "    margin: 0;"
    "}"
    ".login-container {"
    "    background: rgba(255, 255, 255, 0.2);"
    "    padding: 20px;"
    "    border-radius: 10px;"
    "    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);"
    "    text-align: center;"
    "    width: 300px;"
    "}"
    "input[type='text'], input[type='password'] {"
    "    width: 90%;"
    "    padding: 10px;"
    "    margin: 10px 0;"
    "    border: none;"
    "    border-radius: 5px;"
    "    font-size: 16px;"
    "}"
    "input[type='submit'] {"
    "    background: #ff6a00;"
    "    color: white;"
    "    border: none;"
    "    padding: 10px 15px;"
    "    font-size: 16px;"
    "    border-radius: 5px;"
    "    cursor: pointer;"
    "    transition: 0.3s;"
    "}"
    "input[type='submit']:hover {"
    "    background: #e65c00;"
    "}"
    "</style>"
    "</head>"
    "<body>"
    "<div class='login-container'>"
    "    <h2>Login</h2>"
    "    <form action='/stream' method='POST'>"
    "        <input type='text' name='user' placeholder='Username' required><br>"
    "        <input type='password' name='pass' placeholder='Password' required><br>"
    "        <input type='submit' value='Login'>"
    "    </form>"
    "</div>"
    "</body>"
    "</html>";

string streamHeader =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";

// Function to ignore SIGPIPE (prevents crashes when client disconnects)
void ignoreSigpipe() {
    struct sigaction sa;
    sa.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &sa, nullptr);
}

void handleClient(int clientSocket) {
    char buffer[1024] = {0};
    read(clientSocket, buffer, sizeof(buffer));

    string request(buffer);
    if (request.find("POST /stream") != string::npos) {
        if (request.find("user=" + string(USERNAME)) != string::npos &&
            request.find("pass=" + string(PASSWORD)) != string::npos) {
            send(clientSocket, streamHeader.c_str(), streamHeader.length(), 0);

            VideoCapture cap("libcamerasrc ! video/x-raw,width=1280,height=720,framerate=15/1 ! videoconvert ! appsink", cv::CAP_GSTREAMER);
            if (!cap.isOpened()) {
                cerr << "Error: Cannot open camera" << endl;
                close(clientSocket);
                return;
            }

            Mat frame, prevFrame, grayFrame, prevGray, diffFrame;
            vector<uchar> buffer;
            vector<int> params = {IMWRITE_JPEG_QUALITY, 80};

            cap >> prevFrame;
            if (prevFrame.empty()) {
                cerr << "Error: First frame capture failed!" << endl;
                close(clientSocket);
                return;
            }
            cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

            while (true) {
                cap >> frame;
                if (frame.empty()) break;

                cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
                absdiff(prevGray, grayFrame, diffFrame);
                threshold(diffFrame, diffFrame, 50, 255, THRESH_BINARY);

                int motionPixels = countNonZero(diffFrame);
                if (motionPixels > MOTION_THRESHOLD) {
                    cout << "Motion detected!" << endl;
                }

                buffer.clear();
                imencode(".jpg", frame, buffer, params);

                string frameHeader = "--frame\r\n"
                                     "Content-Type: image/jpeg\r\n"
                                     "Content-Length: " + to_string(buffer.size()) + "\r\n\r\n";

                if (send(clientSocket, frameHeader.c_str(), frameHeader.size(), 0) == -1) {
                    cerr << "Client disconnected." << endl;
                    break;
                }
                if (send(clientSocket, reinterpret_cast<const char*>(buffer.data()), buffer.size(), 0) == -1) {
                    cerr << "Client disconnected." << endl;
                    break;
                }
                if (send(clientSocket, "\r\n", 2, 0) == -1) {
                    cerr << "Client disconnected." << endl;
                    break;
                }

                prevGray = grayFrame.clone();
                this_thread::sleep_for(chrono::milliseconds(33));
            }
        }
    } else {
        send(clientSocket, loginPage.c_str(), loginPage.length(), 0);
    }
    close(clientSocket);
}

int main() {
    ignoreSigpipe();  // Ignore SIGPIPE to prevent crashes

    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        cerr << "Error: Cannot create socket" << endl;
        return -1;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        cerr << "Error: Bind failed" << endl;
        return -1;
    }

    if (listen(serverSocket, 5) < 0) {
        cerr << "Error: Listen failed" << endl;
        return -1;
    }

    cout << "Server started on port " << PORT << endl;

    while (true) {
        int clientSocket = accept(serverSocket, nullptr, nullptr);
        if (clientSocket < 0) {
            cerr << "Error: Cannot accept connection" << endl;
            continue;
        }
        thread(handleClient, clientSocket).detach();
    }

    close(serverSocket);
    return 0;
}

