#!/usr/bin/env bash
set -u

container_name="epd_test_container"
showimage_signature="ros2 run image_tools showimage --ros-args --remap /image:=/easy_perception_deployment/output"
stopped_any=0
cleanup_failed=0

echo "[kill] Starting cleanup for easy_perception_deployment."

if sudo docker ps --filter "name=^/${container_name}$" --format '{{.Names}}' | grep -Fxq "${container_name}"; then
  if sudo docker stop "${container_name}" -t 1 >/dev/null; then
    echo "[kill] stopped: Docker container '${container_name}'."
    stopped_any=1
  else
    echo "[kill] partial cleanup: failed to stop Docker container '${container_name}'." >&2
    cleanup_failed=1
  fi
else
  echo "[kill] already stopped: Docker container '${container_name}' is not running."
fi

showimage_pids="$(pgrep -f "${showimage_signature}" || true)"
if [[ -n "${showimage_pids}" ]]; then
  if pkill -f "${showimage_signature}"; then
    echo "[kill] stopped: showimage process(es) for this app."
    stopped_any=1
  else
    echo "[kill] partial cleanup: failed to stop showimage process(es)." >&2
    cleanup_failed=1
  fi
else
  echo "[kill] already stopped: no matching showimage process for this app."
fi

if [[ "${cleanup_failed}" -eq 1 ]]; then
  echo "[kill] STATUS: PARTIAL_CLEANUP" >&2
  exit 2
fi

if [[ "${stopped_any}" -eq 1 ]]; then
  echo "[kill] STATUS: STOPPED"
else
  echo "[kill] STATUS: ALREADY_STOPPED"
fi

exit 0
