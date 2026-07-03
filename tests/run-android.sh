#!/bin/bash
# Cross-compiles the KV-checkpoint bench for Android arm64, ships it + a model to
# the devices, runs `reuse_test bench` on each, and collects results.
#
# The build host (needs the NDK) and the adb host (where the phones are attached)
# can differ. Set ADB_HOST to an ssh target to run adb remotely; leave it empty to
# use local adb.
#
#   # phones on this machine:
#   ./run-android.sh
#   # phones attached to a remote box (build here, adb there):
#   ADB_HOST=dgx ./run-android.sh
#   ./run-android.sh models/LFM2.5-230M-Q4_K_M.gguf 8      # model + n_turns
#   BACKEND=opencl ./run-android.sh                        # GPU build (see below)
#
# Env:
#   ANDROID_NDK   NDK path (default: newest under $ANDROID_HOME/ndk)
#   ADB_HOST      ssh target whose adb sees the devices (default: local adb)
#   REMOTE_DIR    staging dir on the adb host (default: /tmp/rnbench)
#   BACKEND       cpu (default) | opencl
#   DEVICES       space-separated adb serials (default: all connected)
#   ABI           android ABI (default: arm64-v8a)
#   API           android platform level (default: 26)
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-models/mamba-130m-hf-Q3_K_M.gguf}"
N_TURNS="${2:-8}"
BACKEND="${BACKEND:-cpu}"
ABI="${ABI:-arm64-v8a}"
API="${API:-26}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/rnbench}"
DEV_DIR="/data/local/tmp/rnbench"
BUILD_DIR="build-android-${BACKEND}"
RESULTS="results-android"
[ -f "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; }
MODEL_BASE="$(basename "$MODEL")"

# --- locate NDK ---
NDK="${ANDROID_NDK:-}"
if [ -z "$NDK" ]; then
    NDK=$(ls -d "${ANDROID_HOME:-$HOME/Library/Android/sdk}"/ndk/* 2>/dev/null | sort -V | tail -1 || true)
fi
[ -n "$NDK" ] && [ -d "$NDK" ] || { echo "NDK not found; set ANDROID_NDK"; exit 1; }
echo "NDK:     $NDK"
echo "backend: $BACKEND   abi: $ABI   api: $API"

# --- adb runner: local or over ssh ---
if [ -n "${ADB_HOST:-}" ]; then
    ship() { scp -q "$1" "$ADB_HOST:$2"; }
    adbh()  { ssh "$ADB_HOST" "adb $*"; }
    # Run a command on the device. The device command ($2) is single-quoted so its
    # spaces/&&/redirects survive the ssh -> adb shell hops intact.
    dev_run() { ssh "$ADB_HOST" "adb -s $1 shell '$2'"; }
    echo "adb host: $ADB_HOST (remote)"
else
    ship() { mkdir -p "$(dirname "$2")"; cp "$1" "$2"; }
    adbh()  { adb "$@"; }
    dev_run() { adb -s "$1" shell "$2"; }
    echo "adb host: local"
fi

# --- cross-compile ---
CMAKE_ARGS=(
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake"
  -DANDROID_ABI="$ABI" -DANDROID_PLATFORM="android-$API"
  -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release
)
if [ "$BACKEND" = "opencl" ]; then
    # OpenCL headers/loader are vendored under the ndk build by build-opencl.sh;
    # RNLLAMA_ANDROID_OPENCL points the CMake at them. (See tests/README.md.)
    CMAKE_ARGS+=(-DGGML_OPENCL=ON)
fi
echo "== building reuse_test ($BACKEND) =="
cmake -B "$BUILD_DIR" -G Ninja "${CMAKE_ARGS[@]}" >/dev/null
cmake --build "$BUILD_DIR" --target reuse_test -j 8 >/dev/null
STRIP=$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip | head -1)
"$STRIP" -o "$BUILD_DIR/reuse_test.stripped" "$BUILD_DIR/reuse_test"
echo "   $(ls -lh "$BUILD_DIR/reuse_test.stripped" | awk '{print $5}') stripped binary"

# --- stage binary + model on the adb host ---
ship "$BUILD_DIR/reuse_test.stripped" "$REMOTE_DIR/reuse_test"
ship "$MODEL"                         "$REMOTE_DIR/$MODEL_BASE"

# --- enumerate devices ---
if [ -n "${DEVICES:-}" ]; then
    SERIALS="$DEVICES"
else
    SERIALS=$(adbh devices | awk 'NR>1 && $2=="device"{print $1}')
fi
[ -n "$SERIALS" ] || { echo "no devices"; exit 1; }
mkdir -p "$RESULTS"
echo "devices: $SERIALS"

# --- run per device ---
for SER in $SERIALS; do
    MODEL_NAME=$(adbh -s "$SER" shell getprop ro.product.model | tr -d '\r' | tr ' ' '_')
    PLATFORM=$(adbh -s "$SER" shell getprop ro.board.platform | tr -d '\r')
    echo ""
    echo "########## $SER  ($MODEL_NAME / $PLATFORM) ##########"
    adbh -s "$SER" shell "mkdir -p $DEV_DIR"
    adbh -s "$SER" push "$REMOTE_DIR/reuse_test"    "$DEV_DIR/reuse_test" >/dev/null 2>&1
    adbh -s "$SER" push "$REMOTE_DIR/$MODEL_BASE"   "$DEV_DIR/$MODEL_BASE" >/dev/null 2>&1
    adbh -s "$SER" shell "chmod 755 $DEV_DIR/reuse_test"
    OUT="$RESULTS/${BACKEND}-${MODEL_NAME}-${SER}.txt"
    # OpenCL: offload to GPU (NGL) and resolve libOpenCL.so from the device's own
    # vendor driver -- our linked stub only exists to satisfy the linker at build time.
    ENV=""
    [ "$BACKEND" = "opencl" ] && ENV="LD_LIBRARY_PATH=/vendor/lib64:/system/vendor/lib64 RNLLAMA_BENCH_NGL=99 "
    dev_run "$SER" "cd $DEV_DIR; ${ENV}./reuse_test $DEV_DIR/$MODEL_BASE bench $N_TURNS" \
        2>/dev/null | tr -d '\r' | tee "$OUT" | grep -E '^ +[0-9]|overall|blob|BENCH ::' || true
    echo "   -> $OUT"
done

echo ""
echo "== summary (overall speedup per device) =="
for f in "$RESULTS"/${BACKEND}-*.txt; do
    [ -f "$f" ] || continue
    sp=$(sed -n 's/.*overall speedup=\([0-9.]*\).*/\1/p' "$f" | tail -1)
    printf "  %-52s %sx\n" "$(basename "$f")" "${sp:-?}"
done
