#include <iostream>
#include <chrono>
#include <unistd.h>

int main(){

    for(int i=0; i<10; i++)
    {
    auto start = std::chrono::steady_clock::now();
    sleep(2);
    auto end = std::chrono::steady_clock::now();

    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    
    std::cout <<  duration_ns.count() << std::endl;
    std::cout <<  duration_ms.count() << std::endl;
    }
    return 0;
}