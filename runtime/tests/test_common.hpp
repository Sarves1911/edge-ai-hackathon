#pragma once

#include <iostream>

#define CHECK(condition)                                      \
    do {                                                      \
        if (!(condition)) {                                   \
            std::cerr                                         \
                << "CHECK failed: "                           \
                << #condition                                 \
                << " at "                                     \
                << __FILE__                                   \
                << ":"                                        \
                << __LINE__                                   \
                << '\n';                                      \
            return 1;                                         \
        }                                                     \
    } while (false)