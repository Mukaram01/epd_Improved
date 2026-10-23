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

// EPD-8 compiled execution-backend capability probe.
// This reports build capabilities only; it does not claim that a GPU is
// physically present or that the NVIDIA container runtime is healthy.

#include <iostream>

#ifndef EPD_ENABLE_TENSORRT
#define EPD_ENABLE_TENSORRT 0
#endif

#ifndef USE_GPU
#define USE_GPU false
#endif

int main()
{
#if defined(__aarch64__)
  constexpr const char * architecture = "aarch64";
#elif defined(__x86_64__)
  constexpr const char * architecture = "x86_64";
#else
  constexpr const char * architecture = "other";
#endif

  std::cout
    << "{\"schema_version\":\"epd_backend_probe/v1\","
    << "\"architecture\":\"" << architecture << "\","
    << "\"cpu\":true,"
    << "\"cuda\":" << (USE_GPU ? "true" : "false") << ","
    << "\"tensorrt\":"
    << ((USE_GPU && EPD_ENABLE_TENSORRT) ? "true" : "false")
    << "}" << std::endl;
  return 0;
}
