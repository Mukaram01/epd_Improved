// Copyright 2026 Easy Perception Deployment contributors
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

// Bounded, offline benchmark of the production P3OrtBase on saved real camera frames.
// Reports per-stage latency, process CPU time, peak RSS and every proposal above 0.05,
// so configurations can be compared for both speed and identical/equivalent outputs.
#include <sys/resource.h>
#include <jsoncpp/json/json.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "ort_cpp_lib/p3_ort_base.hpp"

namespace
{
double cpu_seconds()
{
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6 + usage.ru_stime.tv_sec +
         usage.ru_stime.tv_usec * 1e-6;
}

double peak_rss_mib()
{
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmHWM:") {
      double kib = 0;
      status >> kib;
      return kib / 1024.0;
    }
  }
  return -1;
}
}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 11) {
    std::cerr <<
      "usage: model labels output.json threads newW newH paddedW paddedH repeats image...\n";
    return 2;
  }
  std::vector<std::string> names;
  std::ifstream labels(argv[2]);
  for (std::string name; std::getline(labels, name); ) {
    names.push_back(name);
  }
  const int threads = std::stoi(argv[4]);
  const int newW = std::stoi(argv[5]), newH = std::stoi(argv[6]);
  const int paddedW = std::stoi(argv[7]), paddedH = std::stoi(argv[8]);
  const int repeats = std::stoi(argv[9]);
  const auto load_start = std::chrono::steady_clock::now();
  Ort::P3OrtBase detector(
    static_cast<float>(newW) / 1280.0F, newW, newH, paddedW, paddedH, names.size(), argv[1],
    boost::none, threads, 1, Ort::SessionExecutionMode::SEQUENTIAL,
    std::vector<std::vector<int64_t>>{{3, paddedH, paddedW}});
  detector.initClassNames(names);
  Json::Value report;
  report["threads"] = threads;
  report["input"] = Json::Value(Json::arrayValue);
  for (int v : {newW, newH, paddedW, paddedH}) {
    report["input"].append(v);
  }
  report["session_load_ms"] = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - load_start).count();
  report["images"] = Json::Value(Json::arrayValue);
  const double cpu0 = cpu_seconds();
  const auto wall0 = std::chrono::steady_clock::now();
  int measured = 0;
  for (int n = 10; n < argc; ++n) {
    const cv::Mat image = cv::imread(argv[n]);
    if (image.empty()) {std::cerr << "unreadable " << argv[n] << "\n"; return 3;}
    Json::Value row;
    row["image"] = argv[n];
    for (int r = -1; r < repeats; ++r) {  // r == -1 is an unrecorded warm-up per image
      const auto start = std::chrono::steady_clock::now();
      const auto out = detector.infer(image);
      const double total = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
      if (r < 0) {continue;}
      const auto t = detector.lastInferenceTiming();
      Json::Value timing;
      timing["total_ms"] = total;
      timing["preprocess_ms"] = t.preprocess_ms;
      timing["session_ms"] = t.session_ms;
      timing["postprocess_ms"] = t.postprocess_ms;
      row["timings"].append(timing);
      ++measured;
      if (r == 0) {
        row["detections"] = Json::Value(Json::arrayValue);
        for (size_t i = 0; i < out.size(); ++i) {
          if (out.scores[i] < 0.05F) {continue;}
          const auto & b = out.bboxes[i];
          Json::Value d;
          d["class"] = out.classIndices[i] < names.size() ? names[out.classIndices[i]] : "?";
          d["score"] = out.scores[i];
          for (int v : b) {
            d["bbox"].append(v);
          }
          cv::Mat resized;
          cv::resize(
            out.masks[i], resized,
            cv::Size(std::max(1, b[2] - b[0]), std::max(1, b[3] - b[1])));
          d["mask_pixels"] = cv::countNonZero(resized > 0.5F);
          row["detections"].append(d);
        }
      }
    }
    report["images"].append(row);
  }
  const double wall =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
  report["measured_inferences"] = measured;
  report["cpu_cores_mean"] = (cpu_seconds() - cpu0) / wall;
  report["peak_rss_mib"] = peak_rss_mib();
  std::ofstream(argv[3]) << report;
  std::cout << "threads=" << threads << " input=" << paddedW << "x" << paddedH
            << " cpu_cores=" << report["cpu_cores_mean"].asDouble()
            << " peak_rss_mib=" << report["peak_rss_mib"].asDouble() << std::endl;
  return 0;
}
