// Copyright 2026 Advanced Remanufacturing and Technology Centre
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

#ifndef ORT_CPP_LIB__ORT_NAME_COMPAT_HPP_
#define ORT_CPP_LIB__ORT_NAME_COMPAT_HPP_

#include <cstddef>
#include <string>

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

namespace Ort
{
namespace compat
{
// Session input/output names across ONNX Runtime versions. ORT 1.13 replaced
// GetInputName/GetOutputName with the *Allocated variants; the vendored
// prebuilt runtime is 1.16. Older runtimes keep the original call sequence.
// A missing name is returned as an empty string.
inline std::string inputName(Ort::Session & session, size_t index, OrtAllocator * allocator)
{
#if ORT_API_VERSION >= 13
  const auto name = session.GetInputNameAllocated(index, allocator);
  return name ? std::string(name.get()) : std::string();
#else
  char * raw = session.GetInputName(index, allocator);
  const std::string name = raw == nullptr ? std::string() : std::string(raw);
  if (raw != nullptr) {
    allocator->Free(allocator, raw);
  }
  return name;
#endif
}

inline std::string outputName(Ort::Session & session, size_t index, OrtAllocator * allocator)
{
#if ORT_API_VERSION >= 13
  const auto name = session.GetOutputNameAllocated(index, allocator);
  return name ? std::string(name.get()) : std::string();
#else
  char * raw = session.GetOutputName(index, allocator);
  const std::string name = raw == nullptr ? std::string() : std::string(raw);
  if (raw != nullptr) {
    allocator->Free(allocator, raw);
  }
  return name;
#endif
}
}  // namespace compat
}  // namespace Ort

#endif  // ORT_CPP_LIB__ORT_NAME_COMPAT_HPP_
