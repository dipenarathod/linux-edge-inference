cmake_minimum_required(VERSION 3.18)
project(EdgeInference CXX) #CXX means this is a C++ project. Use this for faster configuration.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to local ONNX Runtime SDK
set(ONNXRUNTIME_DIR "$ENV{HOME}/onnxruntime")

# Add headers
include_directories(${ONNXRUNTIME_DIR}/include)
# Include the helpers directory so you can write #include "helpers.h"
include_directories(src/helpers) #Because helpers.h is in src/helpers, we need to include that directory.
# This line adds the helpers sub-directort to thheader search path. This allows us to use "include helpers.h" 
# directly in main.cpp without needing to specify the full path.

# Add executable (Include both main.cpp and helpers.cpp here)
add_executable(edge_infer 
    src/main.cpp
    src/helpers/helpers.cpp
)

# Find and link libonnxruntime.so
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_DIR}/lib)

# find_library is significantly more powerful than a basic set() command because it 
# proactively verifies the file exists and abstracts away system differences.

# Validation: find_library actively looks inside ${ONNXRUNTIME_DIR}/lib for the actual file. 
# If the file is missing, the variable remains empty, allowing 
# if(NOT ONNXRUNTIME_LIB) block to safely halt the build with a helpful error message.

# Cross-Platform Matching: You only pass the base name onnxruntime. 
# find_library automatically adapts to look for libonnxruntime.so on Linux/Raspberry Pi, 
# libonnxruntime.dylib on macOS, or onnxruntime.lib on Windows. 
# If you used set(), you would have to hardcode the exact filename, 
# ruining cross-platform compatibility.

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "libonnxruntime.so not found under ${ONNXRUNTIME_DIR}/lib")
endif()

target_link_libraries(edge_infer 
    PRIVATE 
        ${ONNXRUNTIME_LIB}
)

# Define the absolute path to your models folder at compile time
target_compile_definitions(edge_infer 
    PRIVATE 
        MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/models/"
)

# target_compile_definitions takes a value from CMake and physically injects
# it into your C++ source code as a preprocessor macro 
# similar to writing #define MODEL_DIR "/home/.../models/" at the top of your code).

# Set RPATH so executable locates libonnxruntime.so automatically at runtime
set_target_properties(edge_infer PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
)
