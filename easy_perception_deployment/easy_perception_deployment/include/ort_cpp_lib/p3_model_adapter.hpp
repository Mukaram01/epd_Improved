// SPDX-License-Identifier: Apache-2.0
#ifndef ORT_CPP_LIB__P3_MODEL_ADAPTER_HPP_
#define ORT_CPP_LIB__P3_MODEL_ADAPTER_HPP_
#include "ort_cpp_lib/ort_base.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <json/json.h>
#include <limits>
#include <opencv2/opencv.hpp>
#include <stdexcept>
namespace Ort {
// Profiles are declarations by the model owner, never inferred from a graph's names.
class ModelProfile
{
public:
  ModelProfile() = default;
  static ModelProfile rgb01()
  {
    ModelProfile p;
    p.full_image_ = true;
    return p;
  }
  bool fullImage() const { return full_image_; }
  static ModelProfile parse(const Json::Value& root)
  {
    if (!root.isMember("model_profile"))
      return {};
    const auto& p = root["model_profile"];
    if (!p.isObject() || p.size() != 3 || !p["preprocessing"].isString() ||
        !p["mask_layout"].isString() || !p["input_layout"].isString())
      throw std::invalid_argument(
        "model_profile requires preprocessing, mask_layout, input_layout");
    if (p["preprocessing"] == "rgb01" && p["mask_layout"] == "full_image" &&
        p["input_layout"] == "nchw")
      return rgb01();
    if (p["preprocessing"] == "legacy_bgr_mean" && p["mask_layout"] == "roi" &&
        p["input_layout"] == "legacy")
      return {};
    throw std::invalid_argument("Unsupported or ambiguous model_profile");
  }

private:
  bool full_image_ = false;
};
inline size_t
checkedTensorSize(const std::vector<int64_t>& shape, bool allow_zero = true)
{
  size_t size = 1;
  for (auto d : shape) {
    if (d < 0 || (!allow_zero && d == 0) ||
        (d && size > std::numeric_limits<size_t>::max() / sizeof(float) / size_t(d)))
      throw std::invalid_argument("Invalid or overflowing tensor dimensions");
    size *= size_t(d);
  }
  return size;
}
inline std::vector<float>
rgb01Tensor(const cv::Mat& bgr, int width, int height)
{
  if (bgr.type() != CV_8UC3 || bgr.empty() || bgr.cols != width || bgr.rows != height)
    throw std::invalid_argument(
      "RGB01 requires BGR8 frame at the exact model raster; no implicit resize");
  const size_t plane = checkedTensorSize({ height, width }, false);
  std::vector<float> tensor(checkedTensorSize({ 3, height, width }, false));
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c)
        tensor[size_t(c) * plane + size_t(y) * width + x] =
          float(bgr.at<cv::Vec3b>(y, x)[2 - c]) / 255.0f;
  return tensor;
}
inline void
legacyP3Tensor(float* dst,
               const cv::Mat& image,
               int newW,
               int newH,
               int paddedW,
               int paddedH,
               InputTensorLayout layout,
               bool uint8_input,
               const cv::Scalar& mean)
{
  if (!dst || image.empty() || image.type() != CV_8UC3 || newW <= 0 || newH <= 0 ||
      newW > paddedW || newH > paddedH)
    throw std::invalid_argument("Invalid legacy P3 image/resize");
  checkedTensorSize({ 3, paddedH, paddedW }, false);
  cv::Mat tmp;
  cv::resize(image, tmp, cv::Size(newW, newH));
  tmp.convertTo(tmp, CV_32FC3);
  if (!uint8_input)
    tmp -= mean;
  cv::Mat padded(paddedH, paddedW, CV_32FC3, cv::Scalar(0, 0, 0));
  tmp.copyTo(padded(cv::Rect(0, 0, newW, newH)));
  for (int y = 0; y < paddedH; ++y)
    for (int x = 0; x < paddedW; ++x)
      for (int c = 0; c < 3; ++c) {
        const size_t index = layout == InputTensorLayout::NHWC
                               ? (size_t(y) * paddedW + x) * 3 + c
                               : (size_t(c) * paddedH + y) * paddedW + x;
        dst[index] = padded.ptr<float>(y, x)[c];
      }
}
struct TensorContract
{
  std::string name;
  int type;
  std::vector<int64_t> shape;
};
// ONNX TensorProto FLOAT=1, INT64=7. The declared metadata and runtime shapes are both checked.
inline void
validateRgb01Contract(const std::vector<TensorContract>& inputs,
                      const std::vector<TensorContract>& outputs,
                      int width,
                      int height)
{
  if (inputs.size() != 1 || outputs.size() != 4 || inputs[0].name != "images" ||
      inputs[0].type != 1 || inputs[0].shape != std::vector<int64_t>({ 1, 3, height, width }) ||
      width <= 0 || height <= 0)
    throw std::invalid_argument("RGB01 requires images: float32[1,3,H,W] with fixed positive H,W");
  const std::array<std::string, 4> names{ "boxes", "labels", "scores", "masks" };
  const std::array<int, 4> types{ 1, 7, 1, 1 };
  const std::array<size_t, 4> ranks{ 2, 1, 1, 4 };
  int64_t known_n = -1;
  for (size_t i = 0; i < 4; ++i) {
    const auto& t = outputs[i];
    if (t.name != names[i] || t.type != types[i] || t.shape.size() != ranks[i])
      throw std::invalid_argument(
        "RGB01 output names/order/types/ranks do not match export contract");
    for (auto d : t.shape)
      if (d < -1)
        throw std::invalid_argument("Unsupported dimension");
    if (t.shape[0] >= 0) {
      if (known_n >= 0 && known_n != t.shape[0])
        throw std::invalid_argument("Inconsistent detection counts");
      known_n = t.shape[0];
    }
  }
  if (outputs[0].shape[1] != 4 || (outputs[3].shape[1] != 1 && outputs[3].shape[1] != -1) ||
      (outputs[3].shape[2] != -1 && outputs[3].shape[2] != height) ||
      (outputs[3].shape[3] != -1 && outputs[3].shape[3] != width))
    throw std::invalid_argument("RGB01 requires boxes[N,4], masks[N,1,H,W]");
}
struct P3Decoded
{
  std::vector<std::array<float, 4>> boxes;
  std::vector<uint64_t> labels;
  std::vector<float> scores;
  std::vector<cv::Mat> masks;
};
inline P3Decoded
decodeP3(const std::vector<OrtBase::DataOutputType>& out,
         ModelProfile profile,
         int width,
         int height,
         int newW,
         int newH,
         uint16_t classes,
         float confidence,
         bool integer_boxes = true)
{
  if (!std::isfinite(confidence) || confidence < 0 || confidence > 1 || width <= 0 || height <= 0 ||
      newW <= 0 || newH <= 0 || out.size() != 4 || out[1].second.size() != 1)
    throw std::invalid_argument("Invalid P3 runtime contract");
  const int64_t n = out[1].second[0];
  if (n < 0 || out[0].second != std::vector<int64_t>({ n, 4 }) ||
      out[2].second != std::vector<int64_t>({ n }))
    throw std::invalid_argument("Inconsistent boxes/labels/scores");
  const auto& shape = out[3].second;
  const bool rank4 = shape.size() == 4 && shape[1] == 1;
  if ((!rank4 && (profile.fullImage() || shape.size() != 3)) || shape[0] != n)
    throw std::invalid_argument("Invalid mask rank/count/channel");
  const int64_t mh = shape[shape.size() - 2], mw = shape.back();
  if (mh <= 0 || mw <= 0 || mh > std::numeric_limits<int>::max() ||
      mw > std::numeric_limits<int>::max() ||
      (profile.fullImage() && (mh != height || mw != width || newW != width || newH != height)))
    throw std::invalid_argument("Invalid mask raster");
  for (const auto& t : out)
    if (checkedTensorSize(t.second) && !t.first)
      throw std::invalid_argument("Null tensor storage");
  const size_t plane = checkedTensorSize({ mh, mw }, false);
  const auto* labels = reinterpret_cast<const int64_t*>(out[1].first);
  P3Decoded result;
  for (int64_t i = 0; i < n; ++i) {
    const float score = out[2].first[i];
    if (!std::isfinite(score) || score < 0 || score > 1 || labels[i] < 0 || labels[i] >= classes ||
        (profile.fullImage() && labels[i] == 0))
      throw std::invalid_argument("Invalid score or class index");
    std::array<float, 4> box;
    for (size_t j = 0; j < 4; ++j) {
      const float raw = out[0].first[size_t(i) * 4 + j];
      if (!std::isfinite(raw))
        throw std::invalid_argument("Nonfinite box");
      // Preserve legacy float division (including rounding at pixel edges).
      // Clip BEFORE integer conversion to avoid overflow on malformed coordinates.
      const double limit = (j % 2 == 0) ? width : height;
      const double scale = (j % 2 == 0) ? double(newW) / width : double(newH) / height;
      const float legacy_scale = (j % 2 == 0) ? float(newW) / width : float(newH) / height;
      const double projected = profile.fullImage() ? double(raw) / scale : double(raw / legacy_scale);
      double value = std::clamp(projected, 0.0, limit);
      box[j] = (integer_boxes || profile.fullImage()) ? float(int(value)) : float(value);
    }
    if (out[0].first[size_t(i) * 4 + 2] <= out[0].first[size_t(i) * 4] ||
        out[0].first[size_t(i) * 4 + 3] <= out[0].first[size_t(i) * 4 + 1] || box[2] <= box[0] ||
        box[3] <= box[1])
      throw std::invalid_argument("Degenerate or malformed box");
    const float* pixels = out[3].first + size_t(i) * plane;
    for (size_t p = 0; p < plane; ++p)
      if (!std::isfinite(pixels[p]) || pixels[p] < 0 || pixels[p] > 1)
        throw std::invalid_argument("Invalid mask probability");
    if (profile.fullImage() ? score < confidence : score <= confidence)
      continue;
    cv::Mat mask(int(mh), int(mw), CV_32FC1, const_cast<float*>(pixels));
    if (profile.fullImage())
      mask = mask(cv::Rect(int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])));
    result.boxes.push_back(box);
    result.labels.push_back(uint64_t(labels[i]));
    result.scores.push_back(score);
    result.masks.push_back(mask.clone());
  }
  return result;
}
} // namespace Ort
#endif
