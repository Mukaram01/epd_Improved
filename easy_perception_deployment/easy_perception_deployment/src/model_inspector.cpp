// Copyright 2026 Easy Perception Deployment contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

namespace
{
std::string jsonEscape(const std::string & value)
{
  std::ostringstream out;
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          out << "?";
        } else {
          out << ch;
        }
        break;
    }
  }
  return out.str();
}

std::string shapeJson(const std::vector<int64_t> & shape)
{
  std::ostringstream out;
  out << "[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      out << ",";
    }
    out << shape[index];
  }
  out << "]";
  return out.str();
}

std::string tensorInfoJson(
  Ort::Session & session,
  Ort::AllocatorWithDefaultOptions & allocator,
  const size_t index,
  const bool input)
{
  char * raw_name = input ?
    session.GetInputName(index, allocator) :
    session.GetOutputName(index, allocator);
  const std::string name = raw_name == nullptr ? "<unknown>" : raw_name;
  if (raw_name != nullptr) {
    allocator.Free(raw_name);
  }

  try {
    Ort::TypeInfo type_info = input ?
      session.GetInputTypeInfo(index) :
      session.GetOutputTypeInfo(index);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    const auto shape = tensor_info.GetShape();
    const int element_type = static_cast<int>(tensor_info.GetElementType());

    std::ostringstream out;
    out << "{\"name\":\"" << jsonEscape(name) << "\",";
    out << "\"tensor\":true,";
    out << "\"element_type\":" << element_type << ",";
    out << "\"rank\":" << shape.size() << ",";
    out << "\"shape\":" << shapeJson(shape) << "}";
    return out.str();
  } catch (const std::exception & exc) {
    std::ostringstream out;
    out << "{\"name\":\"" << jsonEscape(name) << "\",";
    out << "\"tensor\":false,";
    out << "\"error\":\"" << jsonEscape(exc.what()) << "\"}";
    return out.str();
  }
}

void printInvalid(const std::string & model_path, const std::string & error)
{
  std::cout << "{\"inspector_version\":1,\"valid\":false,";
  std::cout << "\"model_path\":\"" << jsonEscape(model_path) << "\",";
  std::cout << "\"error\":\"" << jsonEscape(error) << "\"}" << std::endl;
}
}  // namespace

int main(int argc, char ** argv)
{
  if (argc != 2) {
    printInvalid("", "Usage: epd_model_inspector <model.onnx>");
    return 2;
  }

  const std::string model_path(argv[1]);

  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EPDModelInspector");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, model_path.c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;

    const size_t input_count = session.GetInputCount();
    const size_t output_count = session.GetOutputCount();

    std::ostringstream out;
    out << "{\"inspector_version\":1,\"valid\":true,";
    out << "\"model_path\":\"" << jsonEscape(model_path) << "\",";
    out << "\"input_count\":" << input_count << ",";
    out << "\"output_count\":" << output_count << ",";

    out << "\"inputs\":[";
    for (size_t index = 0; index < input_count; ++index) {
      if (index != 0) {
        out << ",";
      }
      out << tensorInfoJson(session, allocator, index, true);
    }
    out << "],";

    out << "\"outputs\":[";
    for (size_t index = 0; index < output_count; ++index) {
      if (index != 0) {
        out << ",";
      }
      out << tensorInfoJson(session, allocator, index, false);
    }
    out << "]}";

    std::cout << out.str() << std::endl;
    return 0;
  } catch (const std::exception & exc) {
    printInvalid(model_path, exc.what());
    return 3;
  }
}
