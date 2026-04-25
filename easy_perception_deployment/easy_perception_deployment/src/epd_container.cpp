// Copyright 2022 Advanced Remanufacturing and Technology Centre
// Copyright 2022 ROS-Industrial Consortium Asia Pacific Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <jsoncpp/json/json.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "epd_utils_lib/epd_container.hpp"
#include "epd_utils_lib/usecase_config.hpp"

namespace
{
Ort::SessionExecutionMode parseExecutionMode(
  const Json::Value & obj,
  const std::string & config_path,
  const Ort::SessionExecutionMode default_value)
{
  if (!obj.isMember("execution_mode")) {
    return default_value;
  }

  const Json::Value & execution_mode = obj["execution_mode"];
  if (!execution_mode.isString()) {
    throw std::runtime_error(
      "Config 'execution_mode' must be a string in: " + config_path +
      ". Expected 'sequential' or 'parallel'.");
  }

  std::string mode = execution_mode.asString();
  std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (mode == "sequential") {
    return Ort::SessionExecutionMode::SEQUENTIAL;
  }
  if (mode == "parallel") {
    return Ort::SessionExecutionMode::PARALLEL;
  }

  throw std::runtime_error(
    "Config 'execution_mode' must be 'sequential' or 'parallel' in: " +
    config_path);
}

unsigned int parseColorHistogramMetric(const Json::Value & obj, const std::string & usecase_config_path)
{
  if (!obj.isMember("color_match_histogram_metric")) {
    return EPD::COLOR_HISTOGRAM_CORRELATION;
  }

  const Json::Value & metric_value = obj["color_match_histogram_metric"];
  if (metric_value.isInt()) {
    const int metric = metric_value.asInt();
    if (metric < 0 || metric > 3) {
      throw std::runtime_error(
        "Invalid color_match_histogram_metric in use case config file: " +
        usecase_config_path + ". Expected 0-3."
      );
    }
    return static_cast<unsigned int>(metric);
  }

  if (!metric_value.isString()) {
    throw std::runtime_error(
      "Invalid color_match_histogram_metric type in use case config file: " +
      usecase_config_path + ". Expected string or integer."
    );
  }

  std::string metric = metric_value.asString();
  std::transform(metric.begin(), metric.end(), metric.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (metric == "correlation") {
    return EPD::COLOR_HISTOGRAM_CORRELATION;
  }
  if (metric == "chi-square" || metric == "chisquare" || metric == "chi_square") {
    return EPD::COLOR_HISTOGRAM_CHI_SQUARE;
  }
  if (metric == "intersection") {
    return EPD::COLOR_HISTOGRAM_INTERSECTION;
  }
  if (metric == "bhattacharyya") {
    return EPD::COLOR_HISTOGRAM_BHATTACHARYYA;
  }

  throw std::runtime_error(
    "Invalid color_match_histogram_metric in use case config file: " +
    usecase_config_path +
    ". Expected Correlation, Chi-square, Intersection, or Bhattacharyya."
  );
}

bool clampBboxToImage(
  int left,
  int top,
  int right,
  int bottom,
  const cv::Mat & input_image,
  cv::Rect * clamped_rect)
{
  const int clamped_left = std::clamp(left, 0, input_image.cols);
  const int clamped_top = std::clamp(top, 0, input_image.rows);
  const int clamped_right = std::clamp(right, 0, input_image.cols);
  const int clamped_bottom = std::clamp(bottom, 0, input_image.rows);

  if (clamped_right <= clamped_left || clamped_bottom <= clamped_top) {
    return false;
  }

  *clamped_rect = cv::Rect(
    cv::Point(clamped_left, clamped_top),
    cv::Point(clamped_right, clamped_bottom));
  return true;
}

std::string parseImageTransport(const Json::Value & obj, const std::string & config_path)
{
  if (!obj.isMember("image_transport")) {
    return "raw";
  }

  const Json::Value & transport_value = obj["image_transport"];
  if (!transport_value.isString()) {
    throw std::runtime_error(
      "Invalid image_transport type in config file: " +
      config_path + ". Expected string.");
  }

  std::string transport = transport_value.asString();
  std::transform(transport.begin(), transport.end(), transport.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (transport.empty()) {
    return "raw";
  }

  if (transport == "raw" || transport == "compressed" || transport == "compresseddepth") {
    return transport;
  }

  throw std::runtime_error(
    "Invalid image_transport in config file: " + config_path +
    ". Expected raw, compressed, or compressedDepth.");
}

bool parseBooleanField(
  const Json::Value & obj,
  const std::string & key,
  bool default_value,
  const std::string & config_path)
{
  if (!obj.isMember(key)) {
    return default_value;
  }

  const Json::Value & value = obj[key];
  if (value.isBool()) {
    return value.asBool();
  }
  if (value.isInt()) {
    return value.asInt() != 0;
  }
  if (value.isString()) {
    std::string text = value.asString();
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (text == "true" || text == "1" || text == "yes" || text == "on") {
      return true;
    }
    if (text == "false" || text == "0" || text == "no" || text == "off") {
      return false;
    }
  }

  throw std::runtime_error(
    "Invalid " + key + " value in config file: " + config_path +
    ". Expected boolean.");
}

struct ModelInputInfo
{
  std::string input_name;
  std::vector<int64_t> expected_shape;
  std::vector<int64_t> fed_shape;
  Ort::InputTensorLayout layout;
};

std::string shapeToString(const std::vector<int64_t> & shape)
{
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (i + 1 < shape.size()) {
      oss << ",";
    }
  }
  oss << "]";
  return oss.str();
}

const char * layoutToString(Ort::InputTensorLayout layout)
{
  switch (layout) {
    case Ort::InputTensorLayout::CHW:
      return "CHW";
    case Ort::InputTensorLayout::NCHW:
      return "NCHW";
    case Ort::InputTensorLayout::NHWC:
      return "NHWC";
    default:
      return "UNKNOWN";
  }
}

ModelInputInfo inspectModelInputInfo(
  const std::string & model_path,
  int padded_h,
  int padded_w,
  int img_channel)
{
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EPDInputInspector");
  Ort::SessionOptions options;
  Ort::Session session(env, model_path.c_str(), options);
  Ort::AllocatorWithDefaultOptions allocator;

  if (session.GetInputCount() < 1) {
    throw std::runtime_error("Model has no inputs: " + model_path);
  }

  char * input_name = session.GetInputName(0, allocator);
  std::string resolved_input_name = input_name == nullptr ? std::string("<unknown>") : input_name;
  if (input_name != nullptr) {
    allocator.Free(input_name);
  }

  const auto input_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
  const std::vector<int64_t> model_shape = input_info.GetShape();

  if (model_shape.size() == 3) {
    return ModelInputInfo{
      resolved_input_name,
      model_shape,
      {img_channel, padded_h, padded_w},
      Ort::InputTensorLayout::CHW
    };
  }

  if (model_shape.size() == 4) {
    const int64_t c_at_dim1 = model_shape[1];
    const int64_t c_at_dim3 = model_shape[3];
    const bool is_nhwc = (c_at_dim3 == img_channel) && (c_at_dim1 != img_channel);
    const bool is_nchw = (c_at_dim1 == img_channel) || !is_nhwc;

    if (!is_nhwc && !is_nchw) {
      throw std::runtime_error(
              "Unsupported 4D model input layout for input '" + resolved_input_name +
              "' with shape " + shapeToString(model_shape));
    }

    if (is_nhwc) {
      return ModelInputInfo{
        resolved_input_name,
        model_shape,
        {1, padded_h, padded_w, img_channel},
        Ort::InputTensorLayout::NHWC
      };
    }

    return ModelInputInfo{
      resolved_input_name,
      model_shape,
      {1, img_channel, padded_h, padded_w},
      Ort::InputTensorLayout::NCHW
    };
  }

  throw std::runtime_error(
          "Unsupported model input rank for input '" + resolved_input_name + "': " +
          std::to_string(model_shape.size()) + ". Expected rank 3 (CHW) or 4 (NCHW/NHWC).");
}
}  // namespace


namespace EPD
{

EPDContainer::EPDContainer(void)
{
  hasInitialized = false;
  onlyVisualize = true;
  onlyService = false;
  color_match_histogram_metric = EPD::COLOR_HISTOGRAM_CORRELATION;
  image_transport = "raw";
  publish_detection_segmentation = true;
  confidence_threshold = 0.5f;
  max_detections = 100;

  this->setModelConfigFile();
  this->setPrecisionLevel();
  this->setLabelList();
  this->setUseCaseConfigFile();
}

EPDContainer::~EPDContainer() {}

bool EPDContainer::isInit(void)
{
  return hasInitialized;
}

bool EPDContainer::isVisualize(void)
{
  return onlyVisualize;
}

bool EPDContainer::isService(void)
{
  return onlyService;
}

int EPDContainer::getHeight() {return frame_height;}

int EPDContainer::getWidth() {return frame_width;}

void EPDContainer::setInitBoolean(bool input)
{
  hasInitialized = input;
}

void EPDContainer::setFrameDimension(int input_width, int input_height)
{
  frame_width = input_width;
  frame_height = input_height;
}

void EPDContainer::initORTSessionHandler()
{
  const ResizeParams resize_params = calculateResizeParams();
  const float ratio = resize_params.ratio;
  const int newW = resize_params.resized_width;
  const int newH = resize_params.resized_height;
  const int paddedW = resize_params.padded_width;
  const int paddedH = resize_params.padded_height;

  fprintf(
    stdout,
    "[EPDContainer] Inference resize: ratio=%.4f, resized=%dx%d, tensor=%dx%d "
    "(target_min_side=%d, allow_upscale=%s)\n",
    ratio, newW, newH, paddedW, paddedH, target_min_side, allow_upscale ? "true" : "false");

  const ModelInputInfo model_input_info =
    inspectModelInputInfo(onnx_model_path, paddedH, paddedW, IMG_CHANNEL);

  fprintf(
    stdout,
    "[EPDContainer] Model input: name=%s expected_shape=%s selected_layout=%s fed_shape=%s\n",
    model_input_info.input_name.c_str(),
    shapeToString(model_input_info.expected_shape).c_str(),
    layoutToString(model_input_info.layout),
    shapeToString(model_input_info.fed_shape).c_str());

  switch (precision_level) {
    case 2:
      p2_ort_session = std::make_unique<Ort::P2OrtBase>(
        ratio, newW, newH, paddedW, paddedH,
        classNames.size(),
        onnx_model_path,
        0,
        intra_op_num_threads,
        inter_op_num_threads,
        execution_mode,
        std::vector<std::vector<int64_t>>{model_input_info.fed_shape},
        model_input_info.layout,
        log_model_info
      );
      p2_ort_session->initClassNames(classNames);
      break;
    case 3:
      p3_ort_session = std::make_unique<Ort::P3OrtBase>(
        ratio, newW, newH, paddedW, paddedH,
        classNames.size(),
        onnx_model_path,
        0,
        intra_op_num_threads,
        inter_op_num_threads,
        execution_mode,
        std::vector<std::vector<int64_t>>{model_input_info.fed_shape},
        model_input_info.layout,
        log_model_info
      );
      p3_ort_session->initClassNames(classNames);
      break;
  }
}

EPDContainer::ResizeParams EPDContainer::calculateResizeParams() const
{
  if (frame_width <= 0 || frame_height <= 0) {
    throw std::runtime_error(
            "Invalid frame dimension. Width and height must be positive integers.");
  }

  const int input_min_side = std::min(frame_width, frame_height);
  float ratio = static_cast<float>(target_min_side) / static_cast<float>(input_min_side);
  if (!allow_upscale) {
    ratio = std::min(ratio, 1.0f);
  }

  const int resized_width = static_cast<int>(ratio * frame_width);
  const int resized_height = static_cast<int>(ratio * frame_height);
  const int padded_width = static_cast<int>(((resized_width + 31) / 32) * 32);
  const int padded_height = static_cast<int>(((resized_height + 31) / 32) * 32);

  return ResizeParams{ratio, resized_width, resized_height, padded_width, padded_height};
}

void EPDContainer::setModelConfigFile()
{
  Json::Value obj;
  std::ifstream ifs_1(PATH_TO_SESSION_CONFIG);

  if (!ifs_1.is_open()) {
    throw std::runtime_error(
            std::string("Config file not found: ") + PATH_TO_SESSION_CONFIG);
  }

  Json::CharReaderBuilder reader_builder;
  std::string parse_errors;
  if (!Json::parseFromStream(reader_builder, ifs_1, &obj, &parse_errors)) {
    throw std::runtime_error(
            std::string("Failed to parse config file: ") +
            PATH_TO_SESSION_CONFIG + " Errors: " + parse_errors);
  }

  onnx_model_path = obj["path_to_model"].asString();
  class_label_path = obj["path_to_label_list"].asString();

  std::string visualizeFlag = obj["visualizeFlag"].asString();
  std::transform(
    visualizeFlag.begin(), visualizeFlag.end(), visualizeFlag.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});

  if (visualizeFlag == "visualize") {
    onlyVisualize = true;
  } else {
    if (visualizeFlag != "robot") {
      fprintf(
        stderr,
        "[EPDContainer] WARNING: Unrecognized visualizeFlag '%s' in session_config.json. "
        "Expected 'visualize' or 'robot'. Defaulting to 'robot' (action) mode.\n",
        obj["visualizeFlag"].asString().c_str());
    }
    onlyVisualize = false;
  }

  if (obj.isMember("intra_op_num_threads")) {
    if (!obj["intra_op_num_threads"].isInt()) {
      throw std::runtime_error(
              "Config 'intra_op_num_threads' must be an integer in: " +
              PATH_TO_SESSION_CONFIG);
    }
    const int thread_count = obj["intra_op_num_threads"].asInt();
    if (thread_count > 0) {
      intra_op_num_threads = thread_count;
    } else if (thread_count < 0) {
      throw std::runtime_error(
              "Config 'intra_op_num_threads' must be >= 0 in: " +
              PATH_TO_SESSION_CONFIG);
    }
  }

  if (obj.isMember("inter_op_num_threads")) {
    if (!obj["inter_op_num_threads"].isInt()) {
      throw std::runtime_error(
              "Config 'inter_op_num_threads' must be an integer in: " +
              PATH_TO_SESSION_CONFIG);
    }
    const int thread_count = obj["inter_op_num_threads"].asInt();
    if (thread_count > 0) {
      inter_op_num_threads = thread_count;
    } else if (thread_count < 0) {
      throw std::runtime_error(
              "Config 'inter_op_num_threads' must be >= 0 in: " +
              PATH_TO_SESSION_CONFIG);
    }
  }

  execution_mode = parseExecutionMode(
    obj,
    PATH_TO_SESSION_CONFIG,
    Ort::SessionExecutionMode::SEQUENTIAL);

  image_transport = parseImageTransport(obj, PATH_TO_SESSION_CONFIG);
  publish_detection_segmentation = parseBooleanField(
    obj,
    "publish_detection_segmentation",
    publish_detection_segmentation,
    PATH_TO_SESSION_CONFIG);
  log_model_info = parseBooleanField(
    obj,
    "log_model_info",
    log_model_info,
    PATH_TO_SESSION_CONFIG);
  allow_upscale = parseBooleanField(
    obj,
    "allow_upscale",
    allow_upscale,
    PATH_TO_SESSION_CONFIG);

  if (obj.isMember("target_min_side")) {
    const Json::Value & side = obj["target_min_side"];
    if (!side.isInt()) {
      throw std::runtime_error(
              "Config 'target_min_side' must be an integer in: " +
              PATH_TO_SESSION_CONFIG);
    }
    const int target_val = side.asInt();
    if (target_val <= 0) {
      throw std::runtime_error(
              "Config 'target_min_side' must be > 0 in: " +
              PATH_TO_SESSION_CONFIG);
    }
    target_min_side = target_val;
  }

  if (obj.isMember("confidence_threshold")) {
    const Json::Value & ct = obj["confidence_threshold"];
    if (!ct.isNumeric()) {
      throw std::runtime_error(
              "Config 'confidence_threshold' must be a number in: " +
              PATH_TO_SESSION_CONFIG);
    }
    const float ct_val = ct.asFloat();
    if (ct_val < 0.0f || ct_val > 1.0f) {
      throw std::runtime_error(
              "Config 'confidence_threshold' must be in [0.0, 1.0] in: " +
              PATH_TO_SESSION_CONFIG);
    }
    confidence_threshold = ct_val;
  }

  if (obj.isMember("max_detections")) {
    const Json::Value & md = obj["max_detections"];
    if (!md.isInt()) {
      throw std::runtime_error(
              "Config 'max_detections' must be an integer in: " +
              PATH_TO_SESSION_CONFIG);
    }
    const int md_val = md.asInt();
    if (md_val < 0) {
      throw std::runtime_error(
              "Config 'max_detections' must be >= 0 in: " +
              PATH_TO_SESSION_CONFIG);
    }
    max_detections = md_val;
  }

  ifs_1.close();
}

void EPDContainer::setUseCaseConfigFile()
{
  Json::Value obj;
  const std::string usecase_config_path = PATH_TO_USECASE_CONFIG;
  std::ifstream ifs_1(usecase_config_path);
  if (!ifs_1.is_open()) {
    throw std::runtime_error("Use case config file not found: " + usecase_config_path);
  }

  Json::CharReaderBuilder reader_builder;
  std::string parse_errors;
  if (!Json::parseFromStream(reader_builder, ifs_1, &obj, &parse_errors)) {
    throw std::runtime_error(
      "Failed to parse use case config file: " + usecase_config_path + ". " +
      parse_errors
    );
  }

  if (!obj.isMember("usecase_mode")) {
    throw std::runtime_error(
      "Missing required key 'usecase_mode' in use case config file: " + usecase_config_path
    );
  }

  useCaseMode = obj["usecase_mode"].asInt();

  // Validate the use case mode before processing any mode-specific config.
  if (useCaseMode > 4) {
    throw std::runtime_error("Invalid Use Case.\n");
  }

  // Classification Mode. Do nothing.
  // Counting Mode
  if (useCaseMode == EPD::COUNTING_MODE) {
    if (!obj.isMember("class_list") || obj["class_list"].isNull()) {
      throw std::runtime_error(
        "Missing required key 'class_list' for Counting Mode in use case config file: " +
        usecase_config_path);
    }
    Json::Value class_list = obj["class_list"];
    if (!class_list.isArray() || class_list.empty()) {
      throw std::runtime_error(
        "'class_list' must be a non-empty array for Counting Mode in use case config file: " +
        usecase_config_path);
    }
    for (size_t index = 0; index < class_list.size(); ++index) {
      countClassNames.emplace_back(class_list[static_cast<Json::ArrayIndex>(index)].asString());
    }
  }

  if (useCaseMode == EPD::COLOR_MATCHING_MODE) {
    if (!obj.isMember("path_to_color_template") || obj["path_to_color_template"].isNull()) {
      throw std::runtime_error(
        "Missing required key 'path_to_color_template' for Color Matching Mode in "
        "use case config file: " + usecase_config_path);
    }
    template_color_path = obj["path_to_color_template"].asString();
    if (template_color_path.empty()) {
      throw std::runtime_error(
        "'path_to_color_template' must not be empty in use case config file: " +
        usecase_config_path);
    }
    color_match_histogram_metric = parseColorHistogramMetric(obj, usecase_config_path);
    if (obj.isMember("color_match_threshold")) {
      const Json::Value & cmt = obj["color_match_threshold"];
      if (!cmt.isNumeric()) {
        throw std::runtime_error(
          "Config 'color_match_threshold' must be a number in: " + usecase_config_path);
      }
      const float cmt_val = cmt.asFloat();
      if (cmt_val < 0.0f || cmt_val > 1.0f) {
        throw std::runtime_error(
          "Config 'color_match_threshold' must be in [0.0, 1.0] in: " + usecase_config_path);
      }
      color_match_threshold = cmt_val;
    }
  }

  // Localization Mode
  if (useCaseMode == EPD::LOCALISATION_MODE) {
    // Check if model precision level is not 3.
    // If true, issue critical error and close program.
    if (precision_level != 3) {
      throw std::runtime_error("Please use a Precision-Level 3 ONNX model.");
    }
  }

  // Tracking Mode
  if (useCaseMode == EPD::TRACKING_MODE) {
    // Check if model precision level is not 3.
    // If true, issue critical error and close program.
    if (precision_level != 3) {
      throw std::runtime_error("Please use a Precision-Level 3 ONNX model.");
    }
    if (!obj.isMember("track_type")) {
      throw std::runtime_error(
        "Missing required key 'track_type' in use case config file: " + usecase_config_path
      );
    }
    tracker_type = obj["track_type"].asString();
  }

  ifs_1.close();
}

void EPDContainer::setPrecisionLevel()
{
  std::vector<std::vector<int64_t>> empty_inputShapes;

  Ort::OrtBase ort_session(
    onnx_model_path,
    0,
    intra_op_num_threads,
    inter_op_num_threads,
    execution_mode,
    empty_inputShapes);

  switch (ort_session.getNumOutputs()) {
    case 1:
      precision_level = 1;
      break;
    case 3:
      precision_level = 2;
      break;
    case 4:
      precision_level = 3;
      break;
    default:
      throw std::runtime_error("Invalid Precision Level. Report as GitHub issue.");
  }
}

void EPDContainer::setLabelList()
{
  std::string label;
  std::fstream infile;
  infile.open(class_label_path);

  if (!infile.is_open()) {
    throw std::runtime_error("Label list file not found: " + class_label_path);
  }

  while (std::getline(infile, label)) {
    classNames.emplace_back(label);
  }

  infile.close();

  if (classNames.empty()) {
    throw std::runtime_error(
      "Label list is empty. Please provide at least one class label in: " + class_label_path);
  }
}

cv::Mat EPDContainer::visualize(
  const EPD::EPDObjectDetection result,
  const cv::Mat input_image)
{
  // If zero objects detected, return original input image
  if (result.bboxes.size() == 0) {
    return input_image;
  }

  cv::Scalar oneColor(0.0, 0.0, 255.0, 0.0);
  cv::Mat output_image = input_image.clone();

  bool noMasksFound = false;
  cv::Mat curMask, finalMask;
  if (result.masks.size() == 0) {
    noMasksFound = true;
  }

  for (size_t i = 0; i < result.bboxes.size(); ++i) {
    const int curBbox[] = {
      result.bboxes[i][0],
      result.bboxes[i][1],
      result.bboxes[i][2],
      result.bboxes[i][3]
    };

    if (!noMasksFound) {
      curMask = result.masks[i].clone();
    }

    if (curMask.empty() && !noMasksFound) {
      continue;
    }

    cv::Rect curBoxRect;
    if (!clampBboxToImage(
        curBbox[0], curBbox[1], curBbox[2], curBbox[3], input_image, &curBoxRect))
    {
      continue;
    }

    const cv::Scalar & curColor = oneColor;
    const std::string curLabel =
      result.classIndices[i] >= classNames.size() ?
      std::to_string(result.classIndices[i]) : classNames[result.classIndices[i]];

    cv::rectangle(
      output_image, curBoxRect.tl(),
      curBoxRect.br(), curColor, 2);

    int baseLine = 0;
    cv::Size labelSize =
      cv::getTextSize(curLabel, cv::FONT_HERSHEY_COMPLEX, 0.35, 1, &baseLine);
    cv::rectangle(
      output_image, curBoxRect.tl(),
      cv::Point(
        curBoxRect.x + labelSize.width,
        curBoxRect.y + static_cast<int>(1.3 * labelSize.height)),
      curColor, -1);

    // Visualizing masks
    if (!noMasksFound) {
      cv::resize(curMask, curMask, curBoxRect.size());
      // Assigning masks that exceed the maskThreshold.
      finalMask = (curMask > 0.5);
    }

    // Assigning coloredRoi with the bounding box.
    cv::Mat coloredRoi = (0.3 * curColor + 0.7 * output_image(curBoxRect));
    coloredRoi.convertTo(coloredRoi, CV_8UC3);

    if (!noMasksFound) {
      std::vector<cv::Mat> contours;
      cv::Mat hierarchy, tempFinalMask;
      finalMask.convertTo(tempFinalMask, CV_8U);
      cv::findContours(
        tempFinalMask, contours, hierarchy, cv::RETR_TREE,
        cv::CHAIN_APPROX_SIMPLE);
      cv::drawContours(
        coloredRoi, contours, -1, cv::Scalar(0, 0, 255), 2, cv::LINE_8,
        hierarchy, 100);
    }

    if (!noMasksFound) {
      coloredRoi.copyTo(output_image(curBoxRect), finalMask);
    }

    cv::putText(
      output_image, curLabel,
      cv::Point(curBoxRect.x, curBoxRect.y + labelSize.height),
      cv::FONT_HERSHEY_COMPLEX, 0.35, cv::Scalar(255, 255, 255));
  }

  return output_image;
}

cv::Mat EPDContainer::visualize(
  const EPD::EPDObjectTracking result,
  const cv::Mat input_image)
{
  // If zero objects detected, return original input image
  if (result.objects.size() == 0) {
    return input_image;
  }

  cv::Scalar oneColor(0.0, 0.0, 255.0, 0.0);

  cv::Mat output_image = input_image.clone();
  for (size_t i = 0; i < result.objects.size(); ++i) {
    const unsigned int curBbox[] = {
      result.objects[i].roi.x_offset,
      result.objects[i].roi.y_offset,
      result.objects[i].roi.width + result.objects[i].roi.x_offset,
      result.objects[i].roi.height + result.objects[i].roi.y_offset};
    cv::Mat curMask = result.objects[i].mask.clone();

    if (curMask.empty()) {
      continue;
    }

    cv::Rect curBoxRect;
    if (!clampBboxToImage(
        static_cast<int>(curBbox[0]),
        static_cast<int>(curBbox[1]),
        static_cast<int>(curBbox[2]),
        static_cast<int>(curBbox[3]),
        input_image,
        &curBoxRect))
    {
      continue;
    }

    const cv::Scalar & curColor = oneColor;
    std::string curLabel = result.objects[i].name;

    cv::rectangle(
      output_image,
      curBoxRect.tl(),
      curBoxRect.br(),
      curColor,
      2);

    int baseLine = 0;
    cv::Size labelSize =
      cv::getTextSize(curLabel, cv::FONT_HERSHEY_COMPLEX, 0.35, 1, &baseLine);
    cv::rectangle(
      output_image, curBoxRect.tl(),
      cv::Point(
        curBoxRect.x + labelSize.width,
        curBoxRect.y + static_cast<int>(1.3 * labelSize.height)),
      curColor, -1);

    if (result.object_ids.size() != 0) {
      curLabel = curLabel + "_" + result.object_ids[i];
    }

    // Visualizing masks
    cv::resize(curMask, curMask, curBoxRect.size());
    // Assigning masks that exceed the maskThreshold.
    cv::Mat finalMask = (curMask > 0.5);

    // Assigning coloredRoi with the bounding box.
    cv::Mat coloredRoi = (0.3 * curColor + 0.7 * output_image(curBoxRect));
    coloredRoi.convertTo(coloredRoi, CV_8UC3);

    std::vector<cv::Mat> contours;
    cv::Mat hierarchy, tempFinalMask;
    finalMask.convertTo(tempFinalMask, CV_8U);
    cv::findContours(
      tempFinalMask, contours, hierarchy, cv::RETR_TREE,
      cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(
      coloredRoi, contours, -1, cv::Scalar(0, 0, 255), 2, cv::LINE_8,
      hierarchy, 100);

    coloredRoi.copyTo(output_image(curBoxRect), finalMask);

    cv::putText(
      output_image, curLabel,
      cv::Point(curBoxRect.x, curBoxRect.y + labelSize.height),
      cv::FONT_HERSHEY_COMPLEX, 0.35, cv::Scalar(255, 255, 255));
  }

  return output_image;
}
}  // namespace EPD
