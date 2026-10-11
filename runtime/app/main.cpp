#include <cassert>
#include <iostream>
#include "../include/edge_ai/capture/frame_source.hpp"

void printFrameInfo(FrameSource& src){
    size_t idx = 0;
    std::vector<uint64_t> expected_timestamps = {0, 100000000, 250000000, 300000000};
    while(true)
    {
        auto val = src.nextFrame();
        if(val.has_value()){
            assert(val->frame_id == idx);
            assert(val->timestamp_ns == expected_timestamps[idx]);
            std::cout << val->frame_id << " " << val->timestamp_ns << std::endl;
            idx++;
        }
        else{
            std::cout << "END" << std::endl;
            break;
        }
    }
}

int main(){
    ReplaySource src(ReplayMode::REALTIME);
    
    src.addFrame({1, 2,3,4}, 0);
    src.addFrame({5, 6,7,8}, 100000000);
    src.addFrame({9, 10,11,12}, 250000000);
    src.addFrame({13, 14,15,16}, 300000000);

    printFrameInfo(src);

    return 0;
}