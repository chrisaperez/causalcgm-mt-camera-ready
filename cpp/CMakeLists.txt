cmake_minimum_required(VERSION 3.16)
project(causalcgm_mt_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# ONNX Runtime location.
#
# The pip wheel ships libonnxruntime.dylib but NO C++ headers, so a standalone
# SDK is required. We vendor the release tarball whose version matches the
# venv's onnxruntime exactly (see cpp/vendor/, fetched by scripts/fetch_ort.sh).
#
# Matching versions is not fussiness: causalcgm_mt/export.py validates numerical
# parity against PyTorch using the *Python* ORT build. If the C++ runner linked
# a different version, that parity gate would be certifying a build the runner
# never executes.
#
# Override with:  cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
if(NOT ONNXRUNTIME_ROOT)
  file(GLOB ORT_VENDORED "${CMAKE_SOURCE_DIR}/vendor/onnxruntime-*")
  if(ORT_VENDORED)
    list(GET ORT_VENDORED 0 ONNXRUNTIME_ROOT)
    message(STATUS "Using vendored ONNX Runtime: ${ONNXRUNTIME_ROOT}")
  endif()
endif()

# Warn loudly if the C++ SDK and the Python package disagree.
if(ONNXRUNTIME_ROOT AND EXISTS "${ONNXRUNTIME_ROOT}/VERSION_NUMBER")
  file(READ "${ONNXRUNTIME_ROOT}/VERSION_NUMBER" ORT_CXX_VERSION)
  string(STRIP "${ORT_CXX_VERSION}" ORT_CXX_VERSION)
  execute_process(
    COMMAND "${CMAKE_SOURCE_DIR}/../.venv/bin/python" -c
            "import onnxruntime;print(onnxruntime.__version__)"
    OUTPUT_VARIABLE ORT_PY_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(ORT_PY_VERSION AND NOT ORT_CXX_VERSION STREQUAL ORT_PY_VERSION)
    message(WARNING
      "ONNX Runtime version mismatch: C++ SDK ${ORT_CXX_VERSION}, "
      "Python ${ORT_PY_VERSION}. The export-time parity check validates a "
      "different build than this binary will run.")
  else()
    message(STATUS "ORT version : ${ORT_CXX_VERSION} (matches Python)")
  endif()
endif()

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}/include"
        "${ONNXRUNTIME_ROOT}/capi"
        "${ONNXRUNTIME_ROOT}"
  PATH_SUFFIXES onnxruntime onnxruntime/core/session)

find_library(ORT_LIB NAMES onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}/lib"
        "${ONNXRUNTIME_ROOT}/capi"
        "${ONNXRUNTIME_ROOT}")

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
  message(FATAL_ERROR
    "ONNX Runtime not found.\n"
    "  include: ${ORT_INCLUDE_DIR}\n"
    "  library: ${ORT_LIB}\n"
    "Install the C++ SDK and pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

message(STATUS "ORT headers : ${ORT_INCLUDE_DIR}")
message(STATUS "ORT library : ${ORT_LIB}")

add_executable(causalcgm_mt_runner causalcgm_mt_runner.cpp)
target_include_directories(causalcgm_mt_runner PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(causalcgm_mt_runner PRIVATE "${ORT_LIB}")

# RPATH: relative to the binary, not absolute.
#
# An absolute BUILD_RPATH bakes the checkout's path into the executable, so
# renaming or moving the repository makes the runner die at load time with a
# dyld error -- and the C++ parity gate then fails for a reason that has nothing
# to do with parity. `@loader_path` keeps the vendored ORT resolvable wherever
# the tree ends up, and the absolute paths are kept as a fallback for a
# non-vendored -DONNXRUNTIME_ROOT pointing outside the tree.
file(RELATIVE_PATH _ORT_REL
     "${CMAKE_CURRENT_BINARY_DIR}" "${ONNXRUNTIME_ROOT}")
set_target_properties(causalcgm_mt_runner PROPERTIES
  BUILD_RPATH "@loader_path/${_ORT_REL}/lib;@loader_path/${_ORT_REL}/capi;${ONNXRUNTIME_ROOT}/lib;${ONNXRUNTIME_ROOT}/capi"
  INSTALL_RPATH "@loader_path/${_ORT_REL}/lib;@loader_path/${_ORT_REL}/capi")
