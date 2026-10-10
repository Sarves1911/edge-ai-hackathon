#include <cassert>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <edge_ai/core/frame.hpp>
#include <edge_ai/core/time.hpp>

int main(){
    std::vector<uint8_t> dataA = {1,  2,  3,  4,  200, 200,
                                5,  6,  7,  8,  200, 200,
                                 9, 10, 11, 12,  200, 200};
    
    std::vector<uint8_t> dataB = {13,  14,  15,  16,  200, 200,
                                    17,  18,  19,  20,  200, 200,
                                     21, 22, 23, 24,  200, 200};
    ImageView viewA;
    viewA.data = dataA.data();
    viewA.format = PixelFormat::GRAY8;
    viewA.width = 4;
    viewA.height = 3;
    viewA.stride = 6;

    uint8_t val = 1;
    for(size_t i=0; i<viewA.height; i++){
        size_t start = i*viewA.stride;
        for(size_t j=start; j<start+viewA.width; j++)
        {
            assert(viewA.data[j] == val); 
            std::cout << static_cast<int>(viewA.data[j]) << " ";
            val++;
        }
        std::cout << std::endl;
    }

    Frame frameA;
    frameA.view  = viewA;
    frameA.frame_id = 42;
    frameA.timestamp_ns =  now_monotonic_ns();

    sleep(1);

    ImageView viewB;
    viewB.data = dataB.data();
    viewB.format = PixelFormat::GRAY8;
    viewB.width = 4;
    viewB.height = 3;
    viewB.stride = 6;

    for(size_t i=0; i<viewB.height; i++){
        size_t start = i*viewB.stride;
        for(size_t j=start; j<start+viewB.width; j++)
        {
            assert(viewB.data[j] == val); 
            std::cout << static_cast<int>(viewB.data[j]) << " ";
            val++;
        }
        std::cout << std::endl;
    }

    Frame frameB;
    frameB.view  = viewB;
    frameB.frame_id = 43;
    frameB.timestamp_ns = now_monotonic_ns();

    assert(viewA.width == 4);
    assert(viewA.height == 3);
    assert(viewA.stride == 6);
    assert(viewA.format == PixelFormat::GRAY8);

    assert(viewB.width == 4);
    assert(viewB.height == 3);
    assert(viewB.stride == 6);
    assert(viewB.format == PixelFormat::GRAY8);

    assert(frameB.timestamp_ns > frameA.timestamp_ns);
    std::cout << frameB.timestamp_ns - frameA.timestamp_ns << std::endl;
}