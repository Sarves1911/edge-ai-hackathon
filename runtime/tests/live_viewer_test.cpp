#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <edge_ai/viewer/live_viewer.hpp>

#include "test_common.hpp"

namespace {

std::string request(uint16_t port, const std::string& path){
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if(fd < 0) throw std::runtime_error("test socket failed");

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if(connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0){
        close(fd);
        throw std::runtime_error("test connect failed");
    }

    const std::string wire = "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    size_t offset = 0;
    while(offset < wire.size()){
        const ssize_t written = send(fd, wire.data() + offset, wire.size() - offset, 0);
        if(written <= 0){ close(fd); throw std::runtime_error("test send failed"); }
        offset += static_cast<size_t>(written);
    }

    std::string response;
    std::vector<char> buffer(4096);
    while(true){
        const ssize_t count = recv(fd, buffer.data(), buffer.size(), 0);
        if(count < 0){ close(fd); throw std::runtime_error("test receive failed"); }
        if(count == 0) break;
        response.append(buffer.data(), static_cast<size_t>(count));
    }
    close(fd);
    return response;
}

bool throwsInvalidFps(){
    try{
        LiveViewer viewer(0, 0);
    }
    catch(const std::invalid_argument&){
        return true;
    }
    return false;
}

} // namespace

int main(){
    CHECK(throwsInvalidFps());

    TestImage image = makeImage(32, 24, PixelFormat::BGR24, 5, [](size_t x, size_t y, uint8_t* p){
        p[0] = static_cast<uint8_t>(x*4);
        p[1] = static_cast<uint8_t>(y*7);
        p[2] = 80;
    });

    LiveViewer viewer(0, 30);
    CHECK(viewer.port() != 0);

    ViewerStatus status;
    status.frame_id = 42;
    status.state = "ACTIVE";
    status.action = "RUN_DETECTOR";
    status.reason = "motion \"wake\"";
    status.persons = 1;
    status.detector_calls = 7;
    status.frames = 20;
    status.last_infer_ms = 12.5f;
    status.fps = 28.75;
    status.detections = {{4.0f, 3.0f, 24.0f, 20.0f, 0.91f}};
    status.track_ids = {3};
    viewer.publish(image.view, status);

    const std::string page = request(viewer.port(), "/");
    CHECK(page.find("HTTP/1.1 200 OK") == 0);
    CHECK(page.find("LeanVision") != std::string::npos);
    CHECK(page.find("/frame.jpg") != std::string::npos);

    const std::string state = request(viewer.port(), "/state.json");
    CHECK(state.find("HTTP/1.1 200 OK") == 0);
    CHECK(state.find("\"frame_id\":42") != std::string::npos);
    CHECK(state.find("\"persons\":1") != std::string::npos);
    CHECK(state.find("motion \\\"wake\\\"") != std::string::npos);
    CHECK(state.find("\"track_id\":3") != std::string::npos);

    const std::string frame = request(viewer.port(), "/frame.jpg");
    CHECK(frame.find("HTTP/1.1 200 OK") == 0);
    CHECK(frame.find("Content-Type: image/jpeg") != std::string::npos);
    const size_t body = frame.find("\r\n\r\n");
    CHECK(body != std::string::npos);
    CHECK(frame.size() > body + 8);
    CHECK(static_cast<uint8_t>(frame[body + 4]) == 0xFF);
    CHECK(static_cast<uint8_t>(frame[body + 5]) == 0xD8);
    CHECK(static_cast<uint8_t>(frame[frame.size() - 2]) == 0xFF);
    CHECK(static_cast<uint8_t>(frame[frame.size() - 1]) == 0xD9);

    const std::string missing = request(viewer.port(), "/missing");
    CHECK(missing.find("HTTP/1.1 404 Not Found") == 0);

    // The board camera supplies packed YUYV with possible row padding.
    TestImage yuyv = makeImage(32, 24, PixelFormat::YUYV, 4, [](size_t x, size_t y, uint8_t* p){
        p[0] = static_cast<uint8_t>(48 + (x + y) % 150);
        p[1] = 128;
    });
    LiveViewer camera_viewer(0, 30);
    camera_viewer.publish(yuyv.view, status);
    const std::string camera_frame = request(camera_viewer.port(), "/frame.jpg");
    const size_t camera_body = camera_frame.find("\r\n\r\n");
    CHECK(camera_body != std::string::npos);
    CHECK(camera_frame.size() > camera_body + 8);
    CHECK(static_cast<uint8_t>(camera_frame[camera_body + 4]) == 0xFF);
    CHECK(static_cast<uint8_t>(camera_frame[camera_body + 5]) == 0xD8);

    std::cout << "All live viewer tests passed.\n";
    return 0;
}
