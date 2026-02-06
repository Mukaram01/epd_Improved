cmake_minimum_required(VERSION 3.5)
project(easy_perception_deployment)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

# -----------------------------
# Dependencies
# -----------------------------
find_package(ament_cmake REQUIRED)

find_package(epd_onnxruntime_vendor REQUIRED)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(epd_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)

# FIX: missing Quaternion header comes from tf2
find_package(tf2 REQUIRED)

# JsonCPP libraries - use system jsoncpp directly via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

# Create the jsoncpp::jsoncpp target that the code expects
if(NOT TARGET jsoncpp::jsoncpp)
  add_library(jsoncpp::jsoncpp INTERFACE IMPORTED)
  set_target_properties(jsoncpp::jsoncpp PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${JSONCPP_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${JSONCPP_LIBRARIES}"
  )
endif()

# Include paths
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# Add all custom library headers for compilation.
set(EPD_UTILS
  include/epd_utils_lib/epd_container.cpp
  include/ort_cpp_lib/ort_base.cpp
  include/ort_cpp_lib/p3_ort_base.cpp
  include/ort_cpp_lib/p2_ort_base.cpp
)

# -----------------------------
# Model downloads
# -----------------------------
option(EPD_DOWNLOAD_MODELS "Download pretrained ONNX models at configure time" OFF)
option(EPD_REQUIRE_MODELS "Fail configuration when pretrained ONNX models are missing" ON)

if(NOT EPD_DOWNLOAD_MODELS)
  message(WARNING
    "Network model downloads are disabled by default for ROS 2 Humble buildfarm compatibility. "
    "Set -DEPD_DOWNLOAD_MODELS=ON to enable downloads or pre-download the models manually.")
endif()

set(EPD_MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/data/model")
set(EPD_MODEL_SQUEEZENET "${EPD_MODEL_DIR}/squeezenet1.1-7.onnx")
set(EPD_MODEL_FASTER_RCNN "${EPD_MODEL_DIR}/FasterRCNN-10.onnx")
set(EPD_MODEL_MASK_RCNN "${EPD_MODEL_DIR}/MaskRCNN-10.onnx")

set(EPD_MODEL_SQUEEZENET_URL "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx")
set(EPD_MODEL_FASTER_RCNN_URL "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx")
set(EPD_MODEL_MASK_RCNN_URL "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx")

set(EPD_MODEL_SQUEEZENET_SHA256 "1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088")
set(EPD_MODEL_FASTER_RCNN_SHA256 "dfb81423efbea52e45df242ade64cfca0ba05fae78e00cf0c68a0979987f87eb")
set(EPD_MODEL_MASK_RCNN_SHA256 "a519d8102cb162e78cbf123615aa5a8f3bf9d0fa1dec61a2fbbb42fa3f0e0757")

set(EPD_MISSING_MODELS "")

function(epd_verify_model model_path expected_hash)
  if(EXISTS "${model_path}")
    file(SHA256 "${model_path}" model_hash)
    if(NOT model_hash STREQUAL expected_hash)
      message(FATAL_ERROR
        "Model checksum mismatch for ${model_path}\n"
        "Expected SHA256: ${expected_hash}\n"
        "Actual SHA256:   ${model_hash}\n"
        "Delete the file and re-download to continue.")
    endif()
  endif()
endfunction()

foreach(model_spec
    "${EPD_MODEL_SQUEEZENET}|${EPD_MODEL_SQUEEZENET_URL}|${EPD_MODEL_SQUEEZENET_SHA256}"
    "${EPD_MODEL_FASTER_RCNN}|${EPD_MODEL_FASTER_RCNN_URL}|${EPD_MODEL_FASTER_RCNN_SHA256}"
    "${EPD_MODEL_MASK_RCNN}|${EPD_MODEL_MASK_RCNN_URL}|${EPD_MODEL_MASK_RCNN_SHA256}")
  string(REPLACE "|" ";" model_parts "${model_spec}")
  list(GET model_parts 0 model_path)
  list(GET model_parts 1 model_url)
  list(GET model_parts 2 model_hash)

  if(EXISTS "${model_path}")
    epd_verify_model("${model_path}" "${model_hash}")
  else()
    if(EPD_DOWNLOAD_MODELS)
      message(STATUS "Downloading pretrained ONNX model: ${model_url}")
      file(DOWNLOAD
        "${model_url}"
        "${model_path}"
        EXPECTED_HASH "SHA256=${model_hash}")
    else()
      list(APPEND EPD_MISSING_MODELS "${model_path}")
    endif()
  endif()
endforeach()

set(EPD_MODELS_AVAILABLE TRUE)
if(EPD_MISSING_MODELS)
  string(JOIN "\n  - " EPD_MISSING_MODELS_TEXT ${EPD_MISSING_MODELS})
  if(EPD_REQUIRE_MODELS)
    message(FATAL_ERROR
      "Missing pretrained ONNX models and EPD_DOWNLOAD_MODELS is OFF.\n"
      "Expected files:\n  - ${EPD_MISSING_MODELS_TEXT}\n"
      "To download models manually, run the following commands from the repository root:\n"
      "  mkdir -p ${EPD_MODEL_DIR}\n"
      "  curl -L \"${EPD_MODEL_SQUEEZENET_URL}\" -o \"${EPD_MODEL_SQUEEZENET}\"\n"
      "  curl -L \"${EPD_MODEL_FASTER_RCNN_URL}\" -o \"${EPD_MODEL_FASTER_RCNN}\"\n"
      "  curl -L \"${EPD_MODEL_MASK_RCNN_URL}\" -o \"${EPD_MODEL_MASK_RCNN}\"\n"
      "Alternatively, reconfigure with -DEPD_DOWNLOAD_MODELS=ON to download at configure time.")
  else()
    set(EPD_MODELS_AVAILABLE FALSE)
    message(WARNING
      "Missing pretrained ONNX models and EPD_REQUIRE_MODELS=OFF.\n"
      "Model-dependent features and tests will be disabled.\n"
      "Expected files:\n  - ${EPD_MISSING_MODELS_TEXT}")
  endif()
endif()

if(EPD_MODELS_AVAILABLE)
  add_compile_definitions(EPD_MODELS_AVAILABLE=1)
else()
  add_compile_definitions(EPD_MODELS_AVAILABLE=0)
endif()

# -----------------------------
# CUDA check for local onnxruntime build
# -----------------------------
find_package(CUDA)

if(CUDA_FOUND)
  message(AUTHOR_WARNING "Using [-GPU-].")
  add_definitions(-DUSE_GPU=true)
else()
  message(AUTHOR_WARNING "Your local onnxruntime does not support CUDA. Using [-CPU-] instead.")
  add_definitions(-DUSE_GPU=false)
endif()

if(DEBUG)
  message(AUTHOR_WARNING "Running in DEBUG mode.")
  add_definitions(-DPRINT_MODEL_INFO=true)
else()
  add_definitions(-DPRINT_MODEL_INFO=false)
endif()

add_definitions(-DPATH_TO_PACKAGE="${CMAKE_CURRENT_LIST_DIR}")
add_definitions(${PCL_DEFINITIONS})

# Convenience list of deps shared by most targets in this package
set(EPD_COMMON_DEPS
  epd_onnxruntime_vendor
  OpenCV
  cv_bridge
  image_transport
  tf2
)

# -----------------------------
# Tests
# -----------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Excluding to avoid conflicting Doxygen python code blocks.
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_pep257
    ament_cmake_flake8
    ament_cmake_lint_cmake
  )

  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)

  # Helper to apply common link deps to tests
  macro(epd_configure_test tgt)
    ament_target_dependencies(${tgt} ${EPD_COMMON_DEPS})
    target_include_directories(${tgt} PRIVATE ${PCL_INCLUDE_DIRS})
    target_link_libraries(${tgt} jsoncpp::jsoncpp ${PCL_LIBRARIES})
  endmacro()

  if(EPD_MODELS_AVAILABLE)
    ament_add_gtest(epd_test_init test/test_init.cpp ${EPD_UTILS})
    epd_configure_test(epd_test_init)

    ament_add_gtest(epd_test_invalid_usecase test/test_invalid_usecase.cpp ${EPD_UTILS})
    epd_configure_test(epd_test_invalid_usecase)

    ament_add_gtest(epd_test_P2_counting_action test/p2u1a.cpp ${EPD_UTILS})
    epd_configure_test(epd_test_P2_counting_action)

    ament_add_gtest(epd_test_P2_counting_visualize test/p2u1v.cpp ${EPD_UTILS})
    epd_configure_test(epd_test_P2_counting_visualize)

    ament_add_gtest(epd_test_P2_colormatch_action test/p2u2a.cpp ${EPD_UTILS})
    epd_configure_test(epd_test_P2_colormatch_action)

    ament_add_gtest(epd_test_P3_counting_visualize test/p3u1v.cpp ${EPD_UTILS})
    epd_configure_test(epd_test_P3_counting_visualize)

    ament_add_gtest(epd_test_P3_colormatch_visualize test/p3u2v.cpp ${EPD_UTILS})
    epd_configure_test(epd_test_P3_colormatch_visualize)

    ament_add_gtest(epd_test_P3_colormatch_action test/p3u2a.cpp ${EPD_UTILS})
    epd_configure_test(epd_test_P3_colormatch_action)

    ament_add_gtest(epd_test_P3_localize_action test/p3u3a.cpp ${EPD_UTILS})
    epd_configure_test(epd_test_P3_localize_action)

    ament_add_gtest(epd_test_P3_localize_visualize test/p3u3v.cpp ${EPD_UTILS})
    epd_configure_test(epd_test_P3_localize_visualize)

    ament_add_gtest(epd_test_P3_tracking_visualize test/p3u4v.cpp ${EPD_UTILS})
    epd_configure_test(epd_test_P3_tracking_visualize)

    ament_add_gtest(epd_test_P3_tracking_action test/p3u4a.cpp ${EPD_UTILS})
    epd_configure_test(epd_test_P3_tracking_action)

    ament_add_gtest(epd_test_easy_perception_deployment test/test_easy_perception_deployment.cpp ${EPD_UTILS})
    ament_target_dependencies(epd_test_easy_perception_deployment
      epd_onnxruntime_vendor
      rclcpp
      std_msgs
      sensor_msgs
      image_transport
      epd_msgs
      OpenCV
      cv_bridge
      message_filters
      pcl_conversions
      tf2
    )
    target_include_directories(epd_test_easy_perception_deployment PRIVATE ${PCL_INCLUDE_DIRS})
    target_link_libraries(epd_test_easy_perception_deployment ${PCL_LIBRARIES} jsoncpp::jsoncpp)
  else()
    message(WARNING "Skipping model-dependent tests because pretrained ONNX models are unavailable.")
  endif()

endif()

# -----------------------------
# Main node
# -----------------------------
add_executable(easy_perception_deployment src/main.cpp ${EPD_UTILS})

ament_target_dependencies(
  easy_perception_deployment
  epd_onnxruntime_vendor
  rclcpp
  std_msgs
  sensor_msgs
  image_transport
  geometry_msgs
  epd_msgs
  OpenCV
  cv_bridge
  message_filters
  pcl_conversions
  tf2
)

target_include_directories(easy_perception_deployment PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(easy_perception_deployment ${PCL_LIBRARIES} jsoncpp::jsoncpp)

install(TARGETS
  easy_perception_deployment
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
