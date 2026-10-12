add_executable(
    edge_ai_runtime
    main.cpp
)

target_link_libraries(
    edge_ai_runtime
    PRIVATE
        edge_ai::pipeline
        edge_ai::capture
        edge_ai::detector
        edge_ai::config
        edge_ai::metrics
)

if(EDGE_AI_HAVE_ONNXRUNTIME)
    target_link_libraries(edge_ai_runtime PRIVATE edge_ai::onnx_detector)
    target_compile_definitions(edge_ai_runtime PRIVATE EDGE_AI_HAVE_ONNXRUNTIME)
endif()

if(EDGE_AI_HAVE_V4L2)
    target_compile_definitions(edge_ai_runtime PRIVATE EDGE_AI_HAVE_V4L2)
endif()

install(TARGETS edge_ai_runtime RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
