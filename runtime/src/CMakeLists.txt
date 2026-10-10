add_library(
    core
    core/time.cpp
)

add_library(
    edge_ai::core
    ALIAS core
)


target_compile_features(
    core
    PUBLIC cxx_std_17
)


target_include_directories(
    core
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