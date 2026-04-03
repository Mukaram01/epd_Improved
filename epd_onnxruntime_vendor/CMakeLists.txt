cmake_minimum_required(VERSION 3.13)

project(epd_onnxruntime_vendor)

find_package(ament_cmake REQUIRED)

# Downloads a prebuilt ONNX Runtime release binary instead of building from
# source.  This reduces installation time from ~60-90 minutes to ~2 minutes.
#
# To change the version, update ONNXRUNTIME_VERSION below.
# Prebuilt releases are available at:
#   https://github.com/microsoft/onnxruntime/releases
macro(fetch_onnxruntime)
  set(ONNXRUNTIME_VERSION "1.16.3")

  # Select the correct prebuilt archive for the host architecture.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ONNXRUNTIME_ARCHIVE "onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz")
  else()
    set(ONNXRUNTIME_ARCHIVE "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
  endif()

  set(ONNXRUNTIME_URL
    "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_ARCHIVE}"
  )

  include(ExternalProject)
  ExternalProject_Add(onnxruntime
    URL               ${ONNXRUNTIME_URL}
    # No configure or build step needed - binary is already compiled.
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    # Copy the extracted tree (include/ + lib/) into our install staging dir.
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_directory
        <SOURCE_DIR>
        ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_install
  )

  install(
    DIRECTORY
      ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_install/
    DESTINATION
      ${CMAKE_INSTALL_PREFIX}/opt/epd_onnxruntime_vendor
    USE_SOURCE_PERMISSIONS
  )
endmacro()

fetch_onnxruntime()

if(WIN32)
  ament_environment_hooks(env_hook/epd_onnxruntime_vendor_library_path.bat)
  set(ENV_VAR_NAME "PATH")
  set(ENV_VAR_VALUE "opt\\epd_onnxruntime_vendor\\bin")
else()
  ament_environment_hooks(env_hook/epd_onnxruntime_vendor_library_path.sh)
  if(APPLE)
    set(ENV_VAR_NAME "DYLD_LIBRARY_PATH")
  else()
    set(ENV_VAR_NAME "LD_LIBRARY_PATH")
  endif()
  set(ENV_VAR_VALUE "opt/epd_onnxruntime_vendor/lib")
endif()
ament_environment_hooks(env_hook/epd_onnxruntime_vendor_library_path.dsv.in)

ament_package(
  CONFIG_EXTRAS "epd_onnxruntime_vendor-extras.cmake.in"
)

