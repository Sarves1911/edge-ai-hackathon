add_executable(
    frame_test
    frame_test.cpp
)

target_link_libraries(
    frame_test
    PRIVATE
        edge_ai::core
)

add_test(
    NAME frame_test
    COMMAND frame_test
)


add_executable(
    time_test
    time_test.cpp
)

target_link_libraries(
    time_test
    PRIVATE
        edge_ai::core
)

add_test(
    NAME time_test
    COMMAND time_test
)


add_executable(
    frame_source_test
    frame_source_test.cpp
)

target_link_libraries(
    frame_source_test
    PRIVATE
        edge_ai::core
)

add_test(
    NAME frame_source_test
    COMMAND frame_source_test
)


add_executable(
    replay_source_test
    replay_source_test.cpp
)

target_link_libraries(
    replay_source_test
    PRIVATE
        edge_ai::core
        edge_ai::capture
)

add_executable(
    replay_realtime_test
    replay_realtime_test.cpp
)

target_link_libraries(
    replay_realtime_test
    PRIVATE
        edge_ai::core
        edge_ai::capture
)

set(TEST_VIDEO
    ${CMAKE_CURRENT_SOURCE_DIR}/data/test.mov
)

if(EXISTS ${TEST_VIDEO})
    add_test(
        NAME replay_source_test
        COMMAND replay_source_test ${TEST_VIDEO}
    )
else()
    message(
        STATUS
        "tests/data/test.mov not found; replay_source_test will not be registered with CTest"
    )
endif()


if(EXISTS ${TEST_VIDEO})
    add_test(
        NAME replay_realtime_test
        COMMAND replay_realtime_test ${TEST_VIDEO}
    )
endif()