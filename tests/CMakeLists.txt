cmake_minimum_required(VERSION 3.16)
project(rnllama_tests VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies and compile options
add_definitions(
    -DNDEBUG
    -DO3
    -DLM_GGML_USE_CPU
)

# Platform-specific defines
if(APPLE)
    add_definitions(
        -DLM_GGML_USE_ACCELERATE
        # Metal disabled for tests to avoid shader file dependencies
        # -DLM_GGML_USE_METAL
    )
elseif(UNIX AND NOT APPLE)
    # Linux-specific defines for CPU affinity and scheduling
    add_definitions(-D_GNU_SOURCE)
endif()

# Force generic CPU optimizations to avoid missing assembly functions
add_definitions(-DLM_GGML_CPU_GENERIC)

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cpp)

# Collect source files using glob patterns
file(GLOB MODEL_FILES ${SOURCE_DIR}/models/*.cpp)
file(GLOB GGML_CPU_C_FILES ${SOURCE_DIR}/ggml-cpu/*.c)
file(GLOB GGML_CPU_CPP_FILES ${SOURCE_DIR}/ggml-cpu/*.cpp)
file(GLOB GGML_CPU_AMX_FILES ${SOURCE_DIR}/ggml-cpu/amx/*.cpp)
file(GLOB LLAMA_FILES ${SOURCE_DIR}/llama*.cpp)
file(GLOB MTMD_FILES ${SOURCE_DIR}/tools/mtmd/*.cpp)
file(GLOB MTMD_MODEL_FILES ${SOURCE_DIR}/tools/mtmd/models/*.cpp)
file(GLOB COMMON_FILES ${SOURCE_DIR}/common/*.cpp)
file(GLOB JINJA_FILES ${SOURCE_DIR}/common/jinja/*.cpp)

# Shared source files for all test executables
set(RNLLAMA_COMMON_SOURCES
    # Core GGML files
    ${SOURCE_DIR}/ggml.c
    ${SOURCE_DIR}/ggml-alloc.c
    ${SOURCE_DIR}/ggml-backend.cpp
    ${SOURCE_DIR}/ggml-backend-dl.cpp
    ${SOURCE_DIR}/ggml-backend-meta.cpp
    ${SOURCE_DIR}/ggml-backend-reg.cpp
    ${SOURCE_DIR}/ggml-opt.cpp
    ${SOURCE_DIR}/ggml-threading.cpp
    ${SOURCE_DIR}/ggml-quants.c
    ${SOURCE_DIR}/gguf.cpp

    # GGML CPU files (globbed)
    ${GGML_CPU_C_FILES}
    ${GGML_CPU_CPP_FILES}
    ${GGML_CPU_AMX_FILES}

    # Llama files (globbed)
    ${LLAMA_FILES}
    ${SOURCE_DIR}/unicode-data.cpp
    ${SOURCE_DIR}/unicode.cpp

    # Common utilities
    ${COMMON_FILES}

    # Jinja template engine
    ${JINJA_FILES}

    ${SOURCE_DIR}/anyascii.c

    # Multimodal support (globbed)
    ${MTMD_MODEL_FILES}
    ${MTMD_FILES}

    # React Native llama APIs
    ${SOURCE_DIR}/rn-llama.cpp
    ${SOURCE_DIR}/rn-completion.cpp
    ${SOURCE_DIR}/rn-tts.cpp
    ${SOURCE_DIR}/rn-slot.cpp
    ${SOURCE_DIR}/rn-slot-manager.cpp

    # Model implementations (globbed)
    ${MODEL_FILES}
)

# Compile the (large) shared core exactly once as an OBJECT library; every test
# executable links the same objects, which is byte-equivalent to compiling the
# sources per-target but ~Nx faster to build.
add_library(rnllama_test_core OBJECT ${RNLLAMA_COMMON_SOURCES})
target_include_directories(rnllama_test_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common/jinja
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/ggml-cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/tools/mtmd
)
if(APPLE)
    target_link_libraries(rnllama_test_core INTERFACE
        "-framework Accelerate"
        "-framework Foundation"
    )
elseif(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(rnllama_test_core INTERFACE
        Threads::Threads
        m  # Math library
        dl # Dynamic loading library
    )
    if(ANDROID)
        # llama/ggml route logging through __android_log_* when __ANDROID__ is set.
        target_link_libraries(rnllama_test_core INTERFACE log)
    endif()
endif()

# One test executable = one source file + the shared core.
function(add_rnllama_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE rnllama_test_core)
endfunction()

add_rnllama_test(rnllama_tests           simple_test.cpp)             # basic integration
add_rnllama_test(parallel_decoding_test  parallel_decoding_test.cpp)  # slot manager / queueing
add_rnllama_test(reuse_test              reuse_test.cpp)              # KV reuse + checkpoint (unit + gated)
add_rnllama_test(eval_scenarios_test     eval_scenarios_test.cpp)     # transcript eval harness (A-E)
add_rnllama_test(swa_cache_corruption_test swa_cache_corruption_test.cpp) # KNOWN-RED SWA repro (see README)
