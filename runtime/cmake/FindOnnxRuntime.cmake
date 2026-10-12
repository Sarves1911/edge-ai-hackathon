# Finds ONNX Runtime from ONNXRUNTIME_ROOT, Homebrew, or an extracted release tarball.
#
# Defines OnnxRuntime_FOUND and the imported target OnnxRuntime::OnnxRuntime.

set(_ort_hints
    ${ONNXRUNTIME_ROOT}
    $ENV{ONNXRUNTIME_ROOT}
    /opt/homebrew/opt/onnxruntime
    /usr/local/opt/onnxruntime
    /opt/onnxruntime
    /usr/local
    /usr
)

find_path(
    OnnxRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${_ort_hints}
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
)

find_library(
    OnnxRuntime_LIBRARY
    NAMES onnxruntime
    HINTS ${_ort_hints}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    OnnxRuntime
    REQUIRED_VARS OnnxRuntime_LIBRARY OnnxRuntime_INCLUDE_DIR
)

if(OnnxRuntime_FOUND AND NOT TARGET OnnxRuntime::OnnxRuntime)
    add_library(OnnxRuntime::OnnxRuntime UNKNOWN IMPORTED)
    set_target_properties(
        OnnxRuntime::OnnxRuntime
        PROPERTIES
            IMPORTED_LOCATION "${OnnxRuntime_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntime_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(OnnxRuntime_INCLUDE_DIR OnnxRuntime_LIBRARY)
