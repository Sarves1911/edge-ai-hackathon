# One static library per runtime module, each exported as edge_ai::<module>.
function(edge_ai_add_module name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})

    add_library(${name} ${ARG_SOURCES})
    add_library(edge_ai::${name} ALIAS ${name})

    target_compile_features(${name} PUBLIC cxx_std_17)

    target_include_directories(
        ${name}
        PUBLIC
            "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

    if(ARG_DEPS)
        target_link_libraries(${name} PUBLIC ${ARG_DEPS})
    endif()

    install(
        TARGETS ${name}
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
endfunction()

edge_ai_add_module(
    core
    SOURCES core/time.cpp
)

set(EDGE_AI_CAPTURE_SOURCES capture/replay_source.cpp)
if(EDGE_AI_HAVE_V4L2)
    list(APPEND EDGE_AI_CAPTURE_SOURCES capture/v4l2_source.cpp)
endif()

edge_ai_add_module(
    capture
    SOURCES ${EDGE_AI_CAPTURE_SOURCES}
    DEPS edge_ai::core PkgConfig::FFMPEG
)

edge_ai_add_module(
    preprocess
    SOURCES preprocess/preprocess.cpp
    DEPS edge_ai::core
)

edge_ai_add_module(
    detector
    SOURCES
        detector/postprocess.cpp
        detector/simulated_detector.cpp
        detector/detector_worker.cpp
    DEPS edge_ai::core edge_ai::preprocess Threads::Threads
)
target_link_libraries(detector PRIVATE nlohmann_json)

if(EDGE_AI_HAVE_ONNXRUNTIME)
    edge_ai_add_module(
        onnx_detector
        SOURCES detector/onnx_detector.cpp
        DEPS edge_ai::detector
    )
    target_link_libraries(onnx_detector PRIVATE OnnxRuntime::OnnxRuntime)
endif()

edge_ai_add_module(
    motion
    SOURCES motion/motion_gate.cpp
    DEPS edge_ai::core
)

edge_ai_add_module(
    tracker
    SOURCES tracker/tracker.cpp
    DEPS edge_ai::detector
)

edge_ai_add_module(
    scheduler
    SOURCES scheduler/adaptive_scheduler.cpp
)

edge_ai_add_module(
    metrics
    SOURCES metrics/profiler.cpp
    DEPS edge_ai::core
)
target_link_libraries(metrics PRIVATE nlohmann_json)

edge_ai_add_module(
    config
    SOURCES config/runtime_config.cpp
    DEPS edge_ai::detector edge_ai::motion edge_ai::tracker edge_ai::scheduler
)
target_link_libraries(config PRIVATE nlohmann_json)

edge_ai_add_module(
    pipeline
    SOURCES pipeline/pipeline.cpp
    DEPS edge_ai::config edge_ai::detector edge_ai::motion edge_ai::tracker edge_ai::scheduler edge_ai::metrics edge_ai::capture
)
