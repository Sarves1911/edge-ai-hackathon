add_executable(
    cpp_runtime
    frame_tests.cpp
)

target_link_libraries(
    cpp_runtime
    PRIVATE edge_ai::core
)

target_compile_options(
    cpp_runtime
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

install(
    TARGETS cpp_runtime
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
