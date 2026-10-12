// SPDX-License-Identifier: Apache-2.0
// Offline verification only: no ROS node, subscribers, motion or controllers.
#include "ort_cpp_lib/p3_ort_base.hpp"
#include <fstream>
#include <iostream>
int
main(int argc, char** argv)
{
  try {
    if (argc != 6 && argc != 7)
      throw std::invalid_argument("usage: probe MODEL BGR_IMAGE legacy|rgb01 NUM_CLASSES "
                                  "OUTPUT.json [localization|tracking]");
    const std::string mode = argv[3];
    if (mode != "legacy" && mode != "rgb01")
      throw std::invalid_argument("unknown explicit profile");
    auto profile = mode == "rgb01" ? Ort::ModelProfile::rgb01() : Ort::ModelProfile{};
    cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (image.empty())
      throw std::invalid_argument("unreadable image");
    const auto classes = std::stoul(argv[4]);
    if (classes < 2 || classes > UINT16_MAX)
      throw std::invalid_argument("invalid class count");
    Ort::P3OrtBase session(1,
                           image.cols,
                           image.rows,
                           image.cols,
                           image.rows,
                           uint16_t(classes),
                           argv[1],
                           boost::none,
                           2,
                           1,
                           Ort::SessionExecutionMode::SEQUENTIAL,
                           boost::none,
                           Ort::InputTensorLayout::NCHW,
                           false,
                           profile);
    auto result = session.infer(image);
    Json::Value root;
    root["profile"] = mode;
    root["detections"] = Json::arrayValue;
    for (size_t i = 0; i < result.bboxes.size(); ++i) {
      Json::Value detection;
      for (auto coordinate : result.bboxes[i])
        detection["box"].append(coordinate);
      detection["label"] = Json::UInt64(result.classIndices[i]);
      detection["score"] = result.scores[i];
      detection["mask_width"] = result.masks[i].cols;
      detection["mask_height"] = result.masks[i].rows;
      const std::string path = std::string(argv[5]) + ".mask_" + std::to_string(i) + ".f32";
      detection["mask_path"] = path;
      std::ofstream mask(path, std::ios::binary);
      mask.exceptions(std::ios::failbit | std::ios::badbit);
      mask.write(reinterpret_cast<const char*>(result.masks[i].ptr<float>()),
                 result.masks[i].total() * sizeof(float));
      root["detections"].append(detection);
    }
    if (argc == 7) {
      if (!profile.fullImage())
        throw std::invalid_argument("consumer check requires rgb01");
      cv::Mat depth(image.size(), CV_16UC1, cv::Scalar(900));
      sensor_msgs::msg::CameraInfo camera;
      camera.k = { 500, 0, image.cols / 2.0, 0, 500, image.rows / 2.0, 0, 0, 1 };
      camera.header.frame_id = "offline_contract_fixture";
      auto verify = [&](const auto& localized) {
        size_t j = 0;
        for (size_t i = 0; i < result.bboxes.size(); ++i) {
          if (result.scores[i] < .8f)
            continue;
          if (j >= localized.objects.size())
            throw std::runtime_error("consumer detection count");
          const auto& object = localized.objects[j++];
          const auto& box = result.bboxes[i];
          const cv::Mat expected = result.masks[i] > .5f;
          if (object.roi.x_offset != unsigned(box[0]) || object.roi.y_offset != unsigned(box[1]) ||
              object.roi.width != unsigned(box[2] - box[0]) ||
              object.roi.height != unsigned(box[3] - box[1]) || object.mask.type() != CV_8UC1 ||
              object.mask.size() != expected.size() ||
              cv::norm(object.mask, expected, cv::NORM_INF) != 0)
            throw std::runtime_error("consumer changed exact ROI or strict mask threshold");
        }
        if (j != localized.objects.size())
          throw std::runtime_error("consumer extra detections");
        root["consumer_exact_masks"] = Json::UInt64(j);
      };
      const std::string consumer = argv[6];
      if (consumer == "localization") {
        verify(session.infer(image, depth, camera, 1000, .8f));
      } else if (consumer == "tracking") {
        std::vector<cv::Ptr<cv::Tracker>> trackers;
        std::vector<int> logs;
        std::vector<EPD::LabelledRect2d> rectangles;
        verify(session.infer(image, depth, camera, 1000, "KCF", trackers, logs, rectangles, .8f));
      } else
        throw std::invalid_argument("unknown consumer");
      root["consumer"] = consumer;
    }
    std::ofstream out(argv[5]);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << root;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
