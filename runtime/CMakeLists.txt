cmake_minimum_required(VERSION 3.25)

project(
    edge-ai-hackathon
    VERSION 1.0.0
    DESCRIPTION "A project to simulate basic router packet processing behaviour in c++17"
    LANGUAGES CXX
)


include(GNUInstallDirs)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)
add_subdirectory(tests)

include(CTest)

install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/runtime/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)