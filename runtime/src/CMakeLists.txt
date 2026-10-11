add_library(
    core
    core/time.cpp
)

add_library(
    edge_ai::core
    ALIAS core
)

add_library(
    capture
    capture/replay_source.cpp
)

add_library(
    edge_ai::capture
    ALIAS capture
)


target_compile_features(
    core
    PUBLIC cxx_std_17
)

target_compile_features(
    capture
    PUBLIC cxx_std_17
)

target_link_libraries(capture PUBLIC PkgConfig::FFMPEG)

target_include_directories(
    core
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"   
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_include_directories(
    capture
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"   
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)


install(
    TARGETS core
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

install(
    TARGETS capture
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)