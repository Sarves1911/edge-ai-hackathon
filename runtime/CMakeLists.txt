cmake_minimum_required(VERSION 3.25)

project(
    edge-ai-hackathon
    VERSION 1.0.0
    DESCRIPTION "Camera-gated adaptive person-detection runtime for the Arduino UNO Q"
    LANGUAGES CXX
)

option(EDGE_AI_WITH_ONNXRUNTIME "Build the ONNX Runtime detector backend" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

pkg_check_modules(
    FFMPEG
    REQUIRED
    IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
    libswscale
)

set(EDGE_AI_HAVE_ONNXRUNTIME OFF)
if(EDGE_AI_WITH_ONNXRUNTIME)
    find_package(OnnxRuntime)
    if(OnnxRuntime_FOUND)
        set(EDGE_AI_HAVE_ONNXRUNTIME ON)
    else()
        message(WARNING "ONNX Runtime not found; building with the simulated detector only. Set ONNXRUNTIME_ROOT to enable it.")
    endif()
endif()

set(EDGE_AI_HAVE_V4L2 OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(EDGE_AI_HAVE_V4L2 ON)
endif()

include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header-only JSON library, vendored so board builds work offline.
add_library(nlohmann_json INTERFACE)
target_include_directories(
    nlohmann_json
    SYSTEM INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party>"
)

add_subdirectory(src)
add_subdirectory(app)

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
