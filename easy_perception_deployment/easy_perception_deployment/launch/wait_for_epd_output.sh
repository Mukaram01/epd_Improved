#!/usr/bin/env bash
set -euo pipefail

TOPIC="${EPD_OUTPUT_TOPIC:-/easy_perception_deployment/epd_p2_output}"
TIMEOUT_SECS="${EPD_OUTPUT_TIMEOUT:-30}"

echo "Waiting up to ${TIMEOUT_SECS}s for one message on ${TOPIC}..."

if timeout "${TIMEOUT_SECS}" ros2 topic echo --once "${TOPIC}" >/dev/null; then
  echo "✅ EPD output received on ${TOPIC}"
else
  echo "❌ Timed out waiting for EPD output on ${TOPIC}"
  exit 1
fi
