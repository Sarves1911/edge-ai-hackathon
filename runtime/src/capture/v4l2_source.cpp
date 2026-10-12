#include <edge_ai/capture/v4l2_source.hpp>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <edge_ai/core/time.hpp>

namespace {

int xioctl(int fd, unsigned long request, void* arg){
    int ret;
    do{
        ret = ioctl(fd, request, arg);
    } while(ret == -1 && errno == EINTR);
    return ret;
}

[[noreturn]] void fail(const std::string& what){
    throw std::runtime_error(what + ": " + std::strerror(errno));
}

} // namespace

V4L2Source::V4L2Source(const V4L2Config& config)
    : config_(config)
{
    try{
        fd_ = open(config_.device.c_str(), O_RDWR | O_NONBLOCK);
        if(fd_ < 0) fail("Could not open " + config_.device);

        v4l2_capability cap{};
        if(xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) fail("VIDIOC_QUERYCAP");
        if(!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) throw std::runtime_error(config_.device + " is not a capture device");
        if(!(cap.capabilities & V4L2_CAP_STREAMING)) throw std::runtime_error(config_.device + " does not support streaming I/O");

        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = static_cast<uint32_t>(config_.width);
        fmt.fmt.pix.height = static_cast<uint32_t>(config_.height);
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        if(xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) fail("VIDIOC_S_FMT");
        if(fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) throw std::runtime_error("Camera does not provide YUYV");

        // The driver may adjust the size; use what it actually delivers.
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;
        stride_ = fmt.fmt.pix.bytesperline != 0 ? fmt.fmt.pix.bytesperline : width_*2;

        v4l2_streamparm parm{};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = config_.fps;
        xioctl(fd_, VIDIOC_S_PARM, &parm);   // best effort; not all drivers support it

        v4l2_requestbuffers req{};
        req.count = config_.buffers;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if(xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) fail("VIDIOC_REQBUFS");
        if(req.count < 2) throw std::runtime_error("Camera granted fewer than 2 buffers");

        buffers_.resize(req.count);
        for(uint32_t i=0; i<req.count; i++){
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            if(xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) fail("VIDIOC_QUERYBUF");

            void* start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
            if(start == MAP_FAILED) fail("mmap");
            buffers_[i] = {start, buf.length};

            if(xioctl(fd_, VIDIOC_QBUF, &buf) < 0) fail("VIDIOC_QBUF");
        }

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if(xioctl(fd_, VIDIOC_STREAMON, &type) < 0) fail("VIDIOC_STREAMON");
        streaming_ = true;
    }
    catch(...){
        cleanup();
        throw;
    }
}

V4L2Source::~V4L2Source(){
    cleanup();
}

void V4L2Source::cleanup() noexcept{
    if(fd_ >= 0 && streaming_){
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd_, VIDIOC_STREAMOFF, &type);
        streaming_ = false;
    }
    for(Buffer& b : buffers_){
        if(b.start != nullptr && b.start != MAP_FAILED) munmap(b.start, b.length);
    }
    buffers_.clear();
    if(fd_ >= 0){
        close(fd_);
        fd_ = -1;
    }
}

void V4L2Source::requeuePending(){
    if(!pending_index_.has_value()) return;

    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = *pending_index_;
    pending_index_.reset();
    if(xioctl(fd_, VIDIOC_QBUF, &buf) < 0) fail("VIDIOC_QBUF");
}

std::optional<Frame> V4L2Source::nextFrame(){
    // The previous frame has been consumed: give its buffer back to the driver first.
    requeuePending();

    int timeouts = 0;
    while(true){
        pollfd pfd{fd_, POLLIN, 0};
        const int ret = poll(&pfd, 1, config_.poll_timeout_ms);

        if(ret < 0){
            if(errno == EINTR) continue;
            fail("poll");
        }
        if(ret == 0){
            ++stats_.poll_timeouts;
            if(++timeouts >= config_.max_poll_timeouts) throw std::runtime_error("Camera stopped delivering frames");
            continue;
        }
        if(pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) throw std::runtime_error("Camera device error");

        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if(xioctl(fd_, VIDIOC_DQBUF, &buf) < 0){
            if(errno == EAGAIN) continue;
            fail("VIDIOC_DQBUF");
        }
        pending_index_ = buf.index;

        if(last_sequence_.has_value() && buf.sequence > *last_sequence_ + 1){
            stats_.dropped += buf.sequence - *last_sequence_ - 1;
        }
        last_sequence_ = buf.sequence;

        uint64_t ts_ns;
        if((buf.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC){
            ts_ns = static_cast<uint64_t>(buf.timestamp.tv_sec)*1'000'000'000ULL +
                    static_cast<uint64_t>(buf.timestamp.tv_usec)*1'000ULL;
        }
        else{
            ts_ns = now_monotonic_ns();
        }

        if(stats_.captured == 0) first_ts_ns_ = ts_ns;
        last_ts_ns_ = ts_ns;
        ++stats_.captured;

        Frame frame;
        frame.view = {
            static_cast<const uint8_t*>(buffers_[buf.index].start),
            width_,
            height_,
            stride_,
            PixelFormat::YUYV
        };
        frame.timestamp_ns = ts_ns;
        frame.frame_id = frame_id_++;
        return frame;
    }
}

V4L2Stats V4L2Source::stats() const{
    V4L2Stats out = stats_;
    if(stats_.captured > 1 && last_ts_ns_ > first_ts_ns_){
        out.effective_fps = (stats_.captured - 1) / ((last_ts_ns_ - first_ts_ns_)/1e9);
    }
    return out;
}
