#include <edge_ai/viewer/live_viewer.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

namespace {

constexpr const char* kViewerHtml = R"HTML(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LeanVision Live</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #080b10; color: #f6f8fb; min-height: 100vh; }
    main { width: min(1120px, 100%); margin: auto; padding: 24px; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
    h1 { font-size: clamp(22px, 4vw, 34px); letter-spacing: -.04em; margin: 0; }
    .sub { color: #8d98a7; margin-top: 4px; font-size: 14px; }
    .connection { display: flex; align-items: center; gap: 8px; color: #9ba6b5; font-size: 13px; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #ff5c5c; box-shadow: 0 0 12px #ff5c5c80; }
    .dot.ok { background: #42e58c; box-shadow: 0 0 12px #42e58c80; }
    .stage { position: relative; min-height: 260px; display: grid; place-items: center; overflow: hidden;
      border: 1px solid #202734; border-radius: 18px; background: #11161e; box-shadow: 0 22px 80px #0008; }
    #feed { width: 100%; height: auto; display: block; }
    #waiting { position: absolute; color: #7f8a99; }
    .live { position: absolute; top: 14px; left: 14px; padding: 6px 9px; border-radius: 7px;
      background: #de3341e8; font-weight: 800; font-size: 11px; letter-spacing: .12em; }
    .frame { position: absolute; right: 14px; top: 14px; padding: 6px 9px; border-radius: 7px;
      background: #06080cb8; color: #bdc6d2; font: 12px ui-monospace, monospace; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin-top: 14px; }
    .card { min-height: 86px; padding: 15px 16px; border: 1px solid #202734; border-radius: 13px; background: #11161e; }
    .label { color: #7f8a99; font-size: 11px; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }
    .value { margin-top: 6px; font-size: 23px; font-weight: 750; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .accent { color: #42e58c; }
    .wide { grid-column: span 2; }
    .reason { font-size: 15px; line-height: 1.35; white-space: normal; }
    .detections { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 14px; min-height: 28px; }
    .chip { border: 1px solid #275f43; color: #77efa9; background: #10261a; border-radius: 999px; padding: 5px 9px; font-size: 12px; }
    footer { color: #667181; font-size: 12px; margin-top: 14px; }
    @media (max-width: 720px) { main { padding: 14px; } .grid { grid-template-columns: repeat(2,minmax(0,1fr)); } }
  </style>
</head>
<body>
<main>
  <header>
    <div><h1>LeanVision <span class="accent">Live</span></h1><div class="sub">Adaptive person detection · Arduino UNO Q</div></div>
    <div class="connection"><span id="dot" class="dot"></span><span id="connection">connecting</span></div>
  </header>
  <section class="stage">
    <div id="waiting">Waiting for the first camera frame…</div>
    <img id="feed" alt="Annotated live camera feed">
    <div class="live">LIVE</div><div id="frame" class="frame">FRAME —</div>
  </section>
  <section class="grid">
    <div class="card"><div class="label">Persons</div><div id="persons" class="value accent">—</div></div>
    <div class="card"><div class="label">Pipeline FPS</div><div id="fps" class="value">—</div></div>
    <div class="card"><div class="label">State</div><div id="state" class="value">—</div></div>
    <div class="card"><div class="label">Action</div><div id="action" class="value">—</div></div>
    <div class="card"><div class="label">Detector calls</div><div id="calls" class="value">—</div></div>
    <div class="card"><div class="label">Inference</div><div id="inference" class="value">—</div></div>
    <div class="card wide"><div class="label">Scheduler reason</div><div id="reason" class="value reason">—</div></div>
  </section>
  <div id="detections" class="detections"></div>
  <footer>The colored rectangles are rendered by the C++ runtime before JPEG encoding. This viewer is intended for a trusted demo network.</footer>
</main>
<script>
  const byId = id => document.getElementById(id);
  let oldUrl = null;
  async function refreshFrame() {
    try {
      const response = await fetch('/frame.jpg?t=' + Date.now(), {cache: 'no-store'});
      if (!response.ok) throw new Error('frame unavailable');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      byId('feed').onload = () => {
        byId('waiting').style.display = 'none';
        if (oldUrl) URL.revokeObjectURL(oldUrl);
        oldUrl = url;
      };
      byId('feed').src = url;
      byId('dot').classList.add('ok');
      byId('connection').textContent = 'connected';
    } catch (_) {
      byId('dot').classList.remove('ok');
      byId('connection').textContent = 'waiting for frames';
    } finally {
      setTimeout(refreshFrame, 200);
    }
  }
  async function refreshState() {
    try {
      const s = await (await fetch('/state.json?t=' + Date.now(), {cache: 'no-store'})).json();
      if (!Number.isFinite(s.frame_id)) throw new Error('state unavailable');
      byId('frame').textContent = 'FRAME ' + s.frame_id;
      byId('persons').textContent = s.persons;
      byId('fps').textContent = Number(s.fps).toFixed(1);
      byId('state').textContent = s.state;
      byId('action').textContent = s.action;
      byId('calls').textContent = s.detector_calls + ' / ' + s.frames;
      byId('inference').textContent = Number(s.infer_ms).toFixed(1) + ' ms';
      byId('reason').textContent = s.reason || '—';
      const list = byId('detections'); list.replaceChildren();
      for (const d of (s.detections || [])) {
        const chip = document.createElement('span'); chip.className = 'chip';
        const id = Number.isInteger(d.track_id) ? 'ID ' + d.track_id + ' · ' : '';
        chip.textContent = id + 'person ' + Math.round(d.score * 100) + '%'; list.appendChild(chip);
      }
    } catch (_) {}
    setTimeout(refreshState, 400);
  }
  refreshFrame(); refreshState();
</script>
</body>
</html>)HTML";

std::string ffmpegError(const char* operation, int code){
    std::array<char, AV_ERROR_MAX_STRING_SIZE> message{};
    av_strerror(code, message.data(), message.size());
    return std::string(operation) + ": " + message.data();
}

uint8_t clampByte(int value){
    return static_cast<uint8_t>(std::clamp(value, 0, 255));
}

void yuvToBgr(uint8_t y, uint8_t u, uint8_t v, uint8_t* output){
    const int c = std::max(0, static_cast<int>(y) - 16);
    const int d = static_cast<int>(u) - 128;
    const int e = static_cast<int>(v) - 128;
    output[2] = clampByte((298*c + 409*e + 128) >> 8);
    output[1] = clampByte((298*c - 100*d - 208*e + 128) >> 8);
    output[0] = clampByte((298*c + 516*d + 128) >> 8);
}

std::vector<uint8_t> toBgr(const ImageView& image){
    if(image.data == nullptr || image.width == 0 || image.height == 0){
        throw std::invalid_argument("viewer image must have non-null pixels and positive dimensions");
    }
    if(image.width > static_cast<size_t>(std::numeric_limits<int>::max()) ||
       image.height > static_cast<size_t>(std::numeric_limits<int>::max()) ||
       image.width > std::numeric_limits<size_t>::max()/3 ||
       image.height > std::numeric_limits<size_t>::max()/(image.width*3)){
        throw std::invalid_argument("viewer image dimensions are too large");
    }

    size_t bytes_per_pixel = 3;
    if(image.format == PixelFormat::GRAY8) bytes_per_pixel = 1;
    if(image.format == PixelFormat::YUYV){
        if(image.width % 2 != 0) throw std::invalid_argument("viewer YUYV image width must be even");
        bytes_per_pixel = 2;
    }
    const size_t minimum_stride = image.width*bytes_per_pixel;
    if(image.stride < minimum_stride) throw std::invalid_argument("viewer image stride is too small");

    std::vector<uint8_t> bgr(image.width*image.height*3);
    for(size_t y=0; y<image.height; ++y){
        const uint8_t* source = image.data + y*image.stride;
        uint8_t* target = bgr.data() + y*image.width*3;

        switch(image.format){
            case PixelFormat::BGR24:
                std::memcpy(target, source, image.width*3);
                break;
            case PixelFormat::RGB24:
                for(size_t x=0; x<image.width; ++x){
                    target[3*x + 0] = source[3*x + 2];
                    target[3*x + 1] = source[3*x + 1];
                    target[3*x + 2] = source[3*x + 0];
                }
                break;
            case PixelFormat::GRAY8:
                for(size_t x=0; x<image.width; ++x){
                    target[3*x + 0] = source[x];
                    target[3*x + 1] = source[x];
                    target[3*x + 2] = source[x];
                }
                break;
            case PixelFormat::YUYV:
                for(size_t x=0; x<image.width; x += 2){
                    const uint8_t* pair = source + 2*x;
                    yuvToBgr(pair[0], pair[1], pair[3], target + 3*x);
                    yuvToBgr(pair[2], pair[1], pair[3], target + 3*(x + 1));
                }
                break;
        }
    }
    return bgr;
}

void setPixel(
    std::vector<uint8_t>& bgr,
    size_t width,
    size_t height,
    int x,
    int y,
    const std::array<uint8_t, 3>& color
){
    if(x < 0 || y < 0 || x >= static_cast<int>(width) || y >= static_cast<int>(height)) return;
    uint8_t* pixel = bgr.data() + (static_cast<size_t>(y)*width + static_cast<size_t>(x))*3;
    pixel[0] = color[0]; pixel[1] = color[1]; pixel[2] = color[2];
}

void drawRectangle(
    std::vector<uint8_t>& bgr,
    size_t width,
    size_t height,
    int x1,
    int y1,
    int x2,
    int y2,
    int thickness,
    const std::array<uint8_t, 3>& color
){
    for(int offset=0; offset<thickness; ++offset){
        for(int x=x1-offset; x<=x2+offset; ++x){
            setPixel(bgr, width, height, x, y1-offset, color);
            setPixel(bgr, width, height, x, y2+offset, color);
        }
        for(int y=y1-offset; y<=y2+offset; ++y){
            setPixel(bgr, width, height, x1-offset, y, color);
            setPixel(bgr, width, height, x2+offset, y, color);
        }
    }
}

void annotate(
    std::vector<uint8_t>& bgr,
    size_t width,
    size_t height,
    const std::vector<Detection>& detections,
    const std::vector<int>& track_ids
){
    constexpr std::array<std::array<uint8_t, 3>, 6> colors{{
        {{83, 235, 74}}, {{244, 183, 64}}, {{238, 99, 239}},
        {{84, 210, 255}}, {{255, 128, 67}}, {{126, 240, 208}}
    }};
    const int thickness = std::clamp(static_cast<int>(width/320) + 2, 2, 5);

    for(size_t i=0; i<detections.size(); ++i){
        const Detection& box = detections[i];
        if(!std::isfinite(box.x1) || !std::isfinite(box.y1) ||
           !std::isfinite(box.x2) || !std::isfinite(box.y2)) continue;

        const int x1 = std::clamp(static_cast<int>(std::floor(box.x1)), 0, static_cast<int>(width) - 1);
        const int y1 = std::clamp(static_cast<int>(std::floor(box.y1)), 0, static_cast<int>(height) - 1);
        const int x2 = std::clamp(static_cast<int>(std::ceil(box.x2)), 0, static_cast<int>(width) - 1);
        const int y2 = std::clamp(static_cast<int>(std::ceil(box.y2)), 0, static_cast<int>(height) - 1);
        if(x2 <= x1 || y2 <= y1) continue;

        const int identity = i < track_ids.size() ? track_ids[i] : static_cast<int>(i);
        const uint64_t color_key = identity < 0
            ? static_cast<uint64_t>(-static_cast<int64_t>(identity))
            : static_cast<uint64_t>(identity);
        const auto& color = colors[static_cast<size_t>(color_key % colors.size())];
        drawRectangle(bgr, width, height, x1, y1, x2, y2, thickness + 2, {{0, 0, 0}});
        drawRectangle(bgr, width, height, x1, y1, x2, y2, thickness, color);
    }
}

class JpegEncoder{
    public:
        ~JpegEncoder(){ reset(); }

        std::vector<uint8_t> encode(
            const ImageView& image,
            const std::vector<Detection>& detections,
            const std::vector<int>& track_ids
        ){
            std::vector<uint8_t> bgr = toBgr(image);
            annotate(bgr, image.width, image.height, detections, track_ids);
            ensure(image.width, image.height);

            int result = av_frame_make_writable(frame_);
            if(result < 0) throw std::runtime_error(ffmpegError("av_frame_make_writable", result));

            const uint8_t* input[] = {bgr.data()};
            const int input_stride[] = {static_cast<int>(image.width*3)};
            result = sws_scale(
                sws_, input, input_stride, 0, static_cast<int>(image.height), frame_->data, frame_->linesize
            );
            if(result != static_cast<int>(image.height)) throw std::runtime_error("viewer JPEG color conversion failed");

            frame_->pts = next_pts_++;
            frame_->quality = context_->global_quality;
            result = avcodec_send_frame(context_, frame_);
            if(result < 0) throw std::runtime_error(ffmpegError("avcodec_send_frame", result));

            av_packet_unref(packet_);
            result = avcodec_receive_packet(context_, packet_);
            if(result < 0) throw std::runtime_error(ffmpegError("avcodec_receive_packet", result));

            return {packet_->data, packet_->data + packet_->size};
        }

    private:
        void ensure(size_t width, size_t height){
            if(context_ != nullptr && width_ == width && height_ == height) return;
            reset();

            const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
            if(codec == nullptr) throw std::runtime_error("FFmpeg MJPEG encoder is unavailable");

            context_ = avcodec_alloc_context3(codec);
            if(context_ == nullptr) throw std::runtime_error("could not allocate FFmpeg JPEG encoder");
            context_->width = static_cast<int>(width);
            context_->height = static_cast<int>(height);
            context_->pix_fmt = AV_PIX_FMT_YUV420P;
            context_->time_base = AVRational{1, 30};
            context_->framerate = AVRational{30, 1};
            context_->color_range = AVCOL_RANGE_JPEG;
            context_->thread_count = 1;
            context_->flags |= AV_CODEC_FLAG_QSCALE;
            context_->global_quality = FF_QP2LAMBDA*5;

            int result = avcodec_open2(context_, codec, nullptr);
            if(result < 0){
                const std::string message = ffmpegError("avcodec_open2", result);
                reset();
                throw std::runtime_error(message);
            }

            frame_ = av_frame_alloc();
            if(frame_ == nullptr){ reset(); throw std::runtime_error("could not allocate FFmpeg JPEG frame"); }
            frame_->format = context_->pix_fmt;
            frame_->width = context_->width;
            frame_->height = context_->height;
            frame_->color_range = AVCOL_RANGE_JPEG;
            result = av_frame_get_buffer(frame_, 32);
            if(result < 0){
                const std::string message = ffmpegError("av_frame_get_buffer", result);
                reset();
                throw std::runtime_error(message);
            }

            packet_ = av_packet_alloc();
            if(packet_ == nullptr){ reset(); throw std::runtime_error("could not allocate FFmpeg JPEG packet"); }

            sws_ = sws_getContext(
                context_->width, context_->height, AV_PIX_FMT_BGR24,
                context_->width, context_->height, context_->pix_fmt,
                SWS_FAST_BILINEAR, nullptr, nullptr, nullptr
            );
            if(sws_ == nullptr){ reset(); throw std::runtime_error("could not create viewer color converter"); }

            width_ = width;
            height_ = height;
            next_pts_ = 0;
        }

        void reset(){
            if(sws_ != nullptr) sws_freeContext(sws_);
            sws_ = nullptr;
            av_packet_free(&packet_);
            av_frame_free(&frame_);
            avcodec_free_context(&context_);
            width_ = 0;
            height_ = 0;
        }

        AVCodecContext* context_ = nullptr;
        AVFrame* frame_ = nullptr;
        AVPacket* packet_ = nullptr;
        SwsContext* sws_ = nullptr;
        size_t width_ = 0;
        size_t height_ = 0;
        int64_t next_pts_ = 0;
};

std::string jsonEscape(const std::string& value){
    std::ostringstream out;
    for(const unsigned char c : value){
        switch(c){
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if(c < 0x20){
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
                        << std::dec << std::setfill(' ');
                }
                else out << static_cast<char>(c);
        }
    }
    return out.str();
}

double finiteOrZero(double value){
    return std::isfinite(value) ? value : 0.0;
}

std::string statusJson(const ViewerStatus& status){
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "{\"frame_id\":" << status.frame_id
        << ",\"state\":\"" << jsonEscape(status.state) << '"'
        << ",\"action\":\"" << jsonEscape(status.action) << '"'
        << ",\"reason\":\"" << jsonEscape(status.reason) << '"'
        << ",\"persons\":" << status.persons
        << ",\"detector_calls\":" << status.detector_calls
        << ",\"frames\":" << status.frames
        << ",\"result_age_ms\":" << finiteOrZero(status.last_result_age_ms)
        << ",\"infer_ms\":" << finiteOrZero(status.last_infer_ms)
        << ",\"motion_score\":" << finiteOrZero(status.motion_score)
        << ",\"fps\":" << finiteOrZero(status.fps)
        << ",\"detections\":[";

    bool first = true;
    for(size_t i=0; i<status.detections.size(); ++i){
        const Detection& box = status.detections[i];
        if(!std::isfinite(box.x1) || !std::isfinite(box.y1) || !std::isfinite(box.x2) ||
           !std::isfinite(box.y2) || !std::isfinite(box.score)) continue;
        if(!first) out << ',';
        first = false;
        out << "{\"x1\":" << box.x1 << ",\"y1\":" << box.y1
            << ",\"x2\":" << box.x2 << ",\"y2\":" << box.y2
            << ",\"score\":" << box.score;
        if(i < status.track_ids.size()) out << ",\"track_id\":" << status.track_ids[i];
        out << '}';
    }
    out << "]}";
    return out.str();
}

void configureClientSocket(int socket_fd){
    timeval timeout{};
    timeout.tv_sec = 2;
    setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#ifdef SO_NOSIGPIPE
    int enabled = 1;
    setsockopt(socket_fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled));
#endif
}

bool sendAll(int socket_fd, const uint8_t* data, size_t size){
    while(size > 0){
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        const ssize_t sent = send(socket_fd, data, size, flags);
        if(sent < 0){
            if(errno == EINTR) continue;
            return false;
        }
        if(sent == 0) return false;
        data += static_cast<size_t>(sent);
        size -= static_cast<size_t>(sent);
    }
    return true;
}

void respond(
    int socket_fd,
    const char* status,
    const char* content_type,
    const uint8_t* body,
    size_t body_size
){
    std::ostringstream headers;
    headers << "HTTP/1.1 " << status << "\r\n"
            << "Content-Type: " << content_type << "\r\n"
            << "Content-Length: " << body_size << "\r\n"
            << "Cache-Control: no-store, no-cache, must-revalidate\r\n"
            << "X-Content-Type-Options: nosniff\r\n"
            << "Connection: close\r\n\r\n";
    const std::string header_text = headers.str();
    if(!sendAll(socket_fd, reinterpret_cast<const uint8_t*>(header_text.data()), header_text.size())) return;
    if(body_size > 0) sendAll(socket_fd, body, body_size);
}

void respondText(int socket_fd, const char* status, const char* content_type, const std::string& body){
    respond(socket_fd, status, content_type, reinterpret_cast<const uint8_t*>(body.data()), body.size());
}

} // namespace

struct LiveViewer::Impl{
    Impl(uint16_t requested_port, uint32_t requested_fps)
        : max_fps(requested_fps)
    {
        if(max_fps == 0 || max_fps > 30) throw std::invalid_argument("viewer FPS must be in [1, 30]");

        listen_fd = socket(AF_INET, SOCK_STREAM, 0);
        if(listen_fd < 0) throw std::runtime_error(std::string("viewer socket: ") + std::strerror(errno));

        int reuse = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        address.sin_port = htons(requested_port);
        if(bind(listen_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0){
            const std::string message = std::string("viewer bind: ") + std::strerror(errno);
            close(listen_fd); listen_fd = -1;
            throw std::runtime_error(message);
        }
        if(listen(listen_fd, 8) < 0){
            const std::string message = std::string("viewer listen: ") + std::strerror(errno);
            close(listen_fd); listen_fd = -1;
            throw std::runtime_error(message);
        }

        socklen_t length = sizeof(address);
        if(getsockname(listen_fd, reinterpret_cast<sockaddr*>(&address), &length) < 0){
            const std::string message = std::string("viewer getsockname: ") + std::strerror(errno);
            close(listen_fd); listen_fd = -1;
            throw std::runtime_error(message);
        }
        bound_port = ntohs(address.sin_port);
        server = std::thread([this]{ serve(); });
    }

    ~Impl(){
        stopping.store(true);
        if(server.joinable()) server.join();
        if(listen_fd >= 0) close(listen_fd);
    }

    void publish(const ImageView& image, const ViewerStatus& status){
        const auto now = std::chrono::steady_clock::now();
        const auto interval = std::chrono::microseconds(1'000'000/max_fps);
        if(last_publish.time_since_epoch().count() != 0 && now - last_publish < interval) return;

        std::vector<uint8_t> encoded = encoder.encode(image, status.detections, status.track_ids);
        std::string state = statusJson(status);
        {
            std::lock_guard<std::mutex> lock(snapshot_mutex);
            jpeg = std::move(encoded);
            json = std::move(state);
        }
        last_publish = now;
    }

    void serve(){
        while(!stopping.load()){
            pollfd descriptor{};
            descriptor.fd = listen_fd;
            descriptor.events = POLLIN;
            const int ready = poll(&descriptor, 1, 200);
            if(ready < 0){
                if(errno == EINTR) continue;
                break;
            }
            if(ready == 0 || !(descriptor.revents & POLLIN)) continue;

            const int client = accept(listen_fd, nullptr, nullptr);
            if(client < 0){
                if(errno == EINTR) continue;
                if(stopping.load()) break;
                continue;
            }
            configureClientSocket(client);
            handle(client);
            close(client);
        }
    }

    void handle(int client){
        std::array<char, 4096> request{};
        const ssize_t received = recv(client, request.data(), request.size() - 1, 0);
        if(received <= 0) return;

        std::istringstream first_line(std::string(request.data(), static_cast<size_t>(received)));
        std::string method;
        std::string target;
        std::string version;
        first_line >> method >> target >> version;
        if(method != "GET"){
            respondText(client, "405 Method Not Allowed", "text/plain; charset=utf-8", "GET only\n");
            return;
        }
        const size_t query = target.find('?');
        if(query != std::string::npos) target.resize(query);

        if(target == "/" || target == "/index.html"){
            respondText(client, "200 OK", "text/html; charset=utf-8", kViewerHtml);
            return;
        }
        if(target == "/healthz"){
            respondText(client, "200 OK", "text/plain; charset=utf-8", "ok\n");
            return;
        }
        if(target == "/favicon.ico"){
            respond(client, "204 No Content", "image/x-icon", nullptr, 0);
            return;
        }
        if(target == "/state.json"){
            std::string state;
            {
                std::lock_guard<std::mutex> lock(snapshot_mutex);
                state = json;
            }
            respondText(client, "200 OK", "application/json; charset=utf-8", state);
            return;
        }
        if(target == "/frame.jpg"){
            std::vector<uint8_t> frame;
            {
                std::lock_guard<std::mutex> lock(snapshot_mutex);
                frame = jpeg;
            }
            if(frame.empty()){
                respondText(client, "503 Service Unavailable", "text/plain; charset=utf-8", "waiting for frame\n");
            }
            else respond(client, "200 OK", "image/jpeg", frame.data(), frame.size());
            return;
        }
        respondText(client, "404 Not Found", "text/plain; charset=utf-8", "not found\n");
    }

    int listen_fd = -1;
    uint16_t bound_port = 0;
    uint32_t max_fps;
    std::atomic<bool> stopping{false};
    std::thread server;
    std::mutex snapshot_mutex;
    std::vector<uint8_t> jpeg;
    std::string json = "{\"status\":\"waiting\"}";
    std::chrono::steady_clock::time_point last_publish{};
    JpegEncoder encoder;
};

LiveViewer::LiveViewer(uint16_t port, uint32_t max_fps)
    : impl_(std::make_unique<Impl>(port, max_fps))
{
}

LiveViewer::~LiveViewer() = default;

void LiveViewer::publish(const ImageView& image, const ViewerStatus& status){
    impl_->publish(image, status);
}

uint16_t LiveViewer::port() const noexcept{
    return impl_->bound_port;
}
