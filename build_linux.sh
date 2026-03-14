#!/usr/bin/env bash
set -euo pipefail
cmake --preset linux-ninja
cmake --build --preset linux-release
