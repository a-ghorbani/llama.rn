#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# All test executables defined in CMakeLists.txt.
TARGETS=(
    rnllama_tests
    parallel_decoding_test
    reuse_test
    eval_scenarios_test
    swa_cache_corruption_test
)

echo "=== Building llama.rn C++ Tests ==="

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Get the correct SDK path for current Xcode version
if [[ "$OSTYPE" == "darwin"* ]]; then
    SDK_PATH=$(xcrun --show-sdk-path)
    echo "Using macOS SDK: $SDK_PATH"
    CMAKE_OSX_SYSROOT="-DCMAKE_OSX_SYSROOT=$SDK_PATH"
else
    CMAKE_OSX_SYSROOT=""
fi

# Configure
echo "Configuring with CMake..."
cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release $CMAKE_OSX_SYSROOT

# Build every test executable (the shared core is compiled once).
echo ""
echo "Building test executables..."
cmake --build . -j"${JOBS:-4}"

for t in "${TARGETS[@]}"; do
    if [ ! -f "$t" ]; then
        echo "Error: Failed to build $t"
        exit 1
    fi
    echo "✓ $t built successfully"
done

echo ""
echo "=== Build Successful ==="
echo ""
echo "Built executables:"
for t in "${TARGETS[@]}"; do
    echo "  - $t"
done
echo ""
echo "To run the default (tiny-model) suite:"
echo "  ./run_tests.sh"
echo ""
echo "Model-gated suites (see tests/README.md):"
echo "  RNLLAMA_TEST_HYBRID_MODEL=<hybrid/SSM gguf> ./run_tests.sh   # reuse_test checkpoint gates"
echo "  RNLLAMA_TEST_MODEL=<real chat gguf> ./run_tests.sh           # eval_scenarios_test full eval"
echo "  RNLLAMA_TEST_SWA_MODEL=<gemma3 gguf> ./run_tests.sh          # SWA repro (KNOWN-RED)"
echo ""
