#!/usr/bin/env bash
set -euo pipefail
cmake --preset macos-clang
cmake --build --preset macos-release
