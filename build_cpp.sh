#!/bin/bash
# Build the C++ alphago_core module
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Find tools
CMAKE=$(which cmake 2>/dev/null)
if [ -z "$CMAKE" ]; then
    echo "Error: cmake not found. Install with: apt install cmake (Linux) or brew install cmake (macOS)"
    exit 1
fi
PYBIND11_DIR=$(uv run python -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null)
PYTHON_EXE=$(uv run python -c 'import sys; print(sys.executable)' 2>/dev/null)
EXT_SUFFIX=$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))' 2>/dev/null)

echo "CMake:    $CMAKE"
echo "Python:   $PYTHON_EXE"
echo "pybind11: $PYBIND11_DIR"
echo "Suffix:   $EXT_SUFFIX"
echo ""

# Configure
mkdir -p build
cd build
$CMAKE .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DPython_EXECUTABLE="$PYTHON_EXE" \
    -Wno-dev \
    2>&1 | grep -E "^(--|CMake|Configuring|Generating)" || true

# Build
echo ""
echo "Building..."
$CMAKE --build . --config Release -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Copy to project root with correct extension
cd "$SCRIPT_DIR"
cp build/alphago_core*.so "alphago_core${EXT_SUFFIX}" 2>/dev/null || true

echo ""
echo "Built: alphago_core${EXT_SUFFIX}"
echo "Test:  uv run python -c 'import alphago_core; print(\"OK\")'"
