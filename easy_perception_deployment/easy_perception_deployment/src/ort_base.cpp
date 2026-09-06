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

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <utility>
#include <numeric>
#include <iostream>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ort_cpp_lib/ort_base.hpp"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#ifndef EPD_ENABLE_TENSORRT
#define EPD_ENABLE_TENSORRT 0
#endif

#if USE_GPU
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#endif

#if USE_GPU && EPD_ENABLE_TENSORRT
#include "onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"
#endif

template<typename T, template<typename, typename = std::allocator<T>> class Container>
std::ostream & operator<<(std::ostream & os, const Container<T> & container)
{
  using ContainerType = Container<T>;
  for (typename ContainerType::const_iterator it = container.begin();
    it != container.end();
    ++it)
  {
    os << *it << " ";
  }
  return os;
}

namespace
{

std::string normalizedBackend(const char * raw, bool legacy_gpu_requested)
{
  std::string value = raw == nullptr ? "" : std::string(raw);
  std::transform(
    value.begin(), value.end(), value.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  if (value.empty()) {
    return legacy_gpu_requested ? "cuda" : "cpu";
  }
  if (value == "gpu" || value == "nvidia") {
    return "cuda";
  }
  if (value == "trt") {
    return "tensorrt";
  }
  if (value == "default") {
    return "auto";
  }
  if (value != "auto" && value != "cpu" && value != "cuda" && value != "tensorrt") {
    throw std::runtime_error(
            "Unsupported EPD_EXECUTION_BACKEND='" + value +
            "'. Expected auto, cpu, cuda, or tensorrt.");
  }
  return value;
}

size_t resolvedGpuIndex(const boost::optional<size_t> & legacy_index)
{
  const char * env_index = std::getenv("EPD_GPU_INDEX");
  if (env_index != nullptr && env_index[0] != '\0') {
    try {
      const long long parsed = std::stoll(env_index);
      if (parsed < 0) {
        throw std::runtime_error("negative GPU index");
      }
      return static_cast<size_t>(parsed);
    } catch (const std::exception &) {
      throw std::runtime_error(
              "EPD_GPU_INDEX must be a non-negative integer, got '" +
              std::string(env_index) + "'.");
    }
  }
  return legacy_index.is_initialized() ? legacy_index.value() : 0U;
}

std::string ortStatusMessage(OrtStatus * status)
{
  if (status == nullptr) {
    return "";
  }
  const char * text = Ort::GetApi().GetErrorMessage(status);
  const std::string message = text == nullptr ? "unknown ONNX Runtime error" : text;
  Ort::GetApi().ReleaseStatus(status);
  return message;
}

void publishResolvedBackend(const std::string & backend)
{
#if defined(_WIN32)
  _putenv_s("EPD_RESOLVED_EXECUTION_BACKEND", backend.c_str());
#else
  setenv("EPD_RESOLVED_EXECUTION_BACKEND", backend.c_str(), 1);
#endif
}

}  // namespace

namespace Ort
{

class OrtBase::OrtBaseImpl
{
public:
  OrtBaseImpl(
    const std::string & modelPath,
    const boost::optional<size_t> & gpuIdx,
    const boost::optional<int> & intraOpNumThreads,
    const boost::optional<int> & interOpNumThreads,
    const boost::optional<SessionExecutionMode> & executionMode,
    const boost::optional<std::vector<std::vector<int64_t>>> & inputShapes,
    const boost::optional<bool> & logModelInfo);
  ~OrtBaseImpl();

  int getNumOutputs(void);
  bool isInputUint8(int inputIdx) const;
  std::vector<DataOutputType> operator()(const std::vector<float *> & inputData);

private:
  void initSession();
  void initModelInfo();
  void logModelInfo() const;

  Ort::Session m_session;
  Ort::Env m_env;
  Ort::AllocatorWithDefaultOptions m_ortAllocator;

  boost::optional<size_t> m_gpuIdx;
  boost::optional<int> m_intraOpNumThreads;
  boost::optional<int> m_interOpNumThreads;
  boost::optional<SessionExecutionMode> m_executionMode;

  std::vector<char *> m_inputNodeNames;
  std::vector<char *> m_outputNodeNames;
  std::vector<std::vector<int64_t>> m_inputShapes;
  std::vector<std::vector<int64_t>> m_outputShapes;
  std::vector<ONNXTensorElementDataType> m_inputElementTypes;

  std::vector<int64_t> m_inputTensorSizes;
  std::vector<int64_t> m_outputTensorSizes;

  uint8_t m_numInputs;
  uint8_t m_numOutputs;
  std::string m_modelPath;
  bool m_inputShapesProvided = false;
  bool m_logModelInfo = false;
};

OrtBase::OrtBase(
  const std::string & modelPath,
  const boost::optional<size_t> & gpuIdx,
  const boost::optional<int> & intraOpNumThreads,
  const boost::optional<int> & interOpNumThreads,
  const boost::optional<SessionExecutionMode> & executionMode,
  const boost::optional<std::vector<std::vector<int64_t>>> & inputShapes,
  const boost::optional<bool> & logModelInfo)
: base_impl_(
    std::make_unique<OrtBaseImpl>(
      modelPath,
      gpuIdx,
      intraOpNumThreads,
      interOpNumThreads,
      executionMode,
      inputShapes,
      logModelInfo))
{}

OrtBase::~OrtBase() = default;

std::vector<OrtBase::DataOutputType> OrtBase::operator()(const std::vector<float *> & inputImgData)
{
  return this->base_impl_->operator()(inputImgData);
}

int OrtBase::getNumOutputs()
{
  return base_impl_->getNumOutputs();
}

bool OrtBase::isInputUint8(int inputIdx) const
{
  return base_impl_->isInputUint8(inputIdx);
}

bool OrtBase::resolveModelInfoLoggingEnabled(const boost::optional<bool> & logModelInfo)
{
  if (logModelInfo.is_initialized()) {
    return logModelInfo.value();
  }

  const char * envModelInfo = std::getenv("EPD_LOG_MODEL_INFO");
  if (envModelInfo == nullptr) {
    return false;
  }

  std::string envValue(envModelInfo);
  std::transform(
    envValue.begin(), envValue.end(), envValue.begin(),
    [](unsigned char c) {return static_cast<char>(std::tolower(c));});
  return envValue == "1" || envValue == "true" || envValue == "yes" || envValue == "on";
}

OrtBase::OrtBaseImpl::OrtBaseImpl(
  const std::string & modelPath,
  const boost::optional<size_t> & gpuIdx,
  const boost::optional<int> & intraOpNumThreads,
  const boost::optional<int> & interOpNumThreads,
  const boost::optional<SessionExecutionMode> & executionMode,
  const boost::optional<std::vector<std::vector<int64_t>>> & inputShapes,
  const boost::optional<bool> & logModelInfo)
: m_session(nullptr),
  m_env(nullptr),
  m_ortAllocator(),
  m_gpuIdx(gpuIdx),
  m_intraOpNumThreads(intraOpNumThreads),
  m_interOpNumThreads(interOpNumThreads),
  m_executionMode(executionMode),
  m_inputNodeNames(),
  m_outputNodeNames(),
  m_inputShapes(),
  m_outputShapes(),
  m_numInputs(0),
  m_numOutputs(0),
  m_modelPath(modelPath),
  m_logModelInfo(OrtBase::resolveModelInfoLoggingEnabled(logModelInfo))
{
  this->initSession();

  if (inputShapes.is_initialized() && !inputShapes->empty()) {
    m_inputShapesProvided = true;
    m_inputShapes = inputShapes.value();
  }

  this->initModelInfo();
}

OrtBase::OrtBaseImpl::~OrtBaseImpl()
{
  for (auto & elem : this->m_inputNodeNames) {
    free(elem);
    elem = nullptr;
  }
  this->m_inputNodeNames.clear();

  for (auto & elem : this->m_outputNodeNames) {
    free(elem);
    elem = nullptr;
  }
  this->m_outputNodeNames.clear();
}

int OrtBase::OrtBaseImpl::getNumOutputs()
{
  return unsigned(m_numOutputs);
}

bool OrtBase::OrtBaseImpl::isInputUint8(int inputIdx) const
{
  if (inputIdx < 0 || static_cast<size_t>(inputIdx) >= m_inputElementTypes.size()) {
    return false;
  }
  return m_inputElementTypes[inputIdx] == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
}

void OrtBase::OrtBaseImpl::initSession()
{
  m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Ort");
  Ort::SessionOptions sessionOptions;
  constexpr int kDefaultCpuIntraOpNumThreads = 2;
  constexpr int kDefaultCpuInterOpNumThreads = 1;

  const int intra_op_threads = m_intraOpNumThreads.is_initialized() &&
    m_intraOpNumThreads.value() > 0 ?
    m_intraOpNumThreads.value() : kDefaultCpuIntraOpNumThreads;
  const int inter_op_threads = m_interOpNumThreads.is_initialized() &&
    m_interOpNumThreads.value() > 0 ?
    m_interOpNumThreads.value() : kDefaultCpuInterOpNumThreads;
  const SessionExecutionMode execution_mode = m_executionMode.is_initialized() ?
    m_executionMode.value() : SessionExecutionMode::SEQUENTIAL;
  const bool is_parallel_execution = execution_mode == SessionExecutionMode::PARALLEL;

  sessionOptions.SetIntraOpNumThreads(intra_op_threads);
  sessionOptions.SetInterOpNumThreads(inter_op_threads);
  sessionOptions.SetExecutionMode(
    is_parallel_execution ? ExecutionMode::ORT_PARALLEL : ExecutionMode::ORT_SEQUENTIAL);

  const bool legacy_gpu_requested = m_gpuIdx.is_initialized();
  const std::string requested_backend = normalizedBackend(
    std::getenv("EPD_EXECUTION_BACKEND"), legacy_gpu_requested);
  const size_t gpu_index = resolvedGpuIndex(m_gpuIdx);
  std::string resolved_backend = requested_backend;

  if (requested_backend == "auto") {
#if USE_GPU
    const std::string cuda_error = ortStatusMessage(
      OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, gpu_index));
    if (cuda_error.empty()) {
      resolved_backend = "cuda";
    } else {
      std::cerr << "[ORT] AUTO CUDA unavailable: " << cuda_error
                << "; falling back to CPU." << std::endl;
      resolved_backend = "cpu";
    }
#else
    resolved_backend = "cpu";
#endif
  } else if (requested_backend == "cuda") {
#if USE_GPU
    const std::string cuda_error = ortStatusMessage(
      OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, gpu_index));
    if (!cuda_error.empty()) {
      throw std::runtime_error(
              "CUDA backend requested but ONNX Runtime could not initialize it: " +
              cuda_error);
    }
#else
    throw std::runtime_error(
            "CUDA backend requested, but this EPD build has USE_GPU=false. "
            "Build epd_onnxruntime_vendor with CUDA support or choose CPU/auto.");
#endif
  } else if (requested_backend == "tensorrt") {
#if USE_GPU && EPD_ENABLE_TENSORRT
    const std::string tensorrt_error = ortStatusMessage(
      OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, gpu_index));
    if (!tensorrt_error.empty()) {
      throw std::runtime_error(
              "TensorRT backend requested but provider initialization failed: " +
              tensorrt_error);
    }
    // TensorRT may delegate unsupported graph partitions to CUDA. CPU remains
    // the final implicit ONNX Runtime fallback for unsupported operators.
    const std::string cuda_error = ortStatusMessage(
      OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, gpu_index));
    if (!cuda_error.empty()) {
      throw std::runtime_error(
              "TensorRT initialized, but its CUDA fallback provider failed: " +
              cuda_error);
    }
#else
    throw std::runtime_error(
            "TensorRT backend requested, but this EPD build does not include the "
            "TensorRT execution provider. Build the ONNX Runtime vendor with "
            "TensorRT and configure EPD with -DEPD_ENABLE_TENSORRT=ON.");
#endif
  } else {
    resolved_backend = "cpu";
  }

  publishResolvedBackend(resolved_backend);

  std::cout << "[ORT] Session config: backend=" << resolved_backend
            << ", requested_backend=" << requested_backend
            << ", gpu_index=" << gpu_index
            << ", execution_mode="
            << (is_parallel_execution ? "parallel" : "sequential")
            << ", intra_op_num_threads=" << intra_op_threads
            << ", inter_op_num_threads=" << inter_op_threads
            << std::endl;

  sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  m_session = Ort::Session(m_env, m_modelPath.c_str(), sessionOptions);
  m_numInputs = m_session.GetInputCount();

  m_inputNodeNames.reserve(m_numInputs);
  m_inputTensorSizes.reserve(m_numInputs);

  m_numOutputs = m_session.GetOutputCount();

  m_outputNodeNames.reserve(m_numOutputs);
  m_outputTensorSizes.reserve(m_numOutputs);
}

void OrtBase::OrtBaseImpl::initModelInfo()
{
  for (int i = 0; i < m_numInputs; i++) {
    Ort::TypeInfo typeInfo = m_session.GetInputTypeInfo(i);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    m_inputElementTypes.emplace_back(tensorInfo.GetElementType());

    if (!m_inputShapesProvided) {
      m_inputShapes.emplace_back(tensorInfo.GetShape());
    }

    const auto & curInputShape = m_inputShapes[i];

    m_inputTensorSizes.emplace_back(
      std::accumulate(
        std::begin(curInputShape),
        std::end(curInputShape),
        1,
        std::multiplies<int64_t>()));

    char * inputName = m_session.GetInputName(i, m_ortAllocator);
    m_inputNodeNames.emplace_back(strdup(inputName));
    m_ortAllocator.Free(inputName);
  }

  for (int i = 0; i < m_numOutputs; ++i) {
    Ort::TypeInfo typeInfo = m_session.GetOutputTypeInfo(i);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

    m_outputShapes.emplace_back(tensorInfo.GetShape());

    char * outputName = m_session.GetOutputName(i, m_ortAllocator);
    m_outputNodeNames.emplace_back(strdup(outputName));
    m_ortAllocator.Free(outputName);
  }

  #if PRINT_MODEL_INFO
  this->logModelInfo();
  #else
  if (m_logModelInfo) {
    this->logModelInfo();
  }
  #endif
}

void OrtBase::OrtBaseImpl::logModelInfo() const
{
  std::stringstream ss;
  ss << "Model IO info\n";
  ss << "Input count: " << unsigned(m_numInputs) << "\n";
  ss << "Input node names: " << m_inputNodeNames << "\n";
  ss << "Input shapes: " << m_inputShapes << "\n";
  ss << "Output count: " << unsigned(m_numOutputs) << "\n";
  ss << "Output node names: " << m_outputNodeNames << "\n";
  ss << "Output shapes: " << m_outputShapes << "\n";
  std::cout << ss.str();
}

std::vector<OrtBase::DataOutputType> OrtBase::OrtBaseImpl::operator()(
  const std::vector<float *> & inputData)
{
  if (m_numInputs != inputData.size()) {
    throw std::runtime_error("Mismatch size of input data\n");
  }
  // ORT accepts CPU tensors as input even when CUDA/TensorRT providers execute
  // the graph; the provider copies/binds data as required by the session.
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> inputTensors;
  inputTensors.reserve(m_numInputs);
  std::vector<std::vector<uint8_t>> uint8Buffers;
  for (int i = 0; i < m_numInputs; ++i) {
    if (m_inputElementTypes[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      const int64_t tensorSize = m_inputTensorSizes[i];
      uint8Buffers.emplace_back(static_cast<size_t>(tensorSize));
      std::vector<uint8_t> & buf = uint8Buffers.back();
      const float * src = inputData[i];
      for (int64_t j = 0; j < tensorSize; ++j) {
        buf[j] = static_cast<uint8_t>(
          std::clamp(src[j], 0.0f, 255.0f));
      }
      inputTensors.emplace_back(
        std::move(
          Ort::Value::CreateTensor<uint8_t>(
            memoryInfo,
            buf.data(),
            static_cast<size_t>(tensorSize),
            m_inputShapes[i].data(),
            m_inputShapes[i].size())));
    } else {
      inputTensors.emplace_back(
        std::move(
          Ort::Value::CreateTensor<float>(
            memoryInfo,
            const_cast<float *>(inputData[i]),
            m_inputTensorSizes[i],
            m_inputShapes[i].data(),
            m_inputShapes[i].size())));
    }
  }
  auto outputTensors = m_session.Run(
    Ort::RunOptions{nullptr},
    m_inputNodeNames.data(),
    inputTensors.data(),
    m_numInputs,
    m_outputNodeNames.data(),
    m_numOutputs);

  if (outputTensors.size() != m_numOutputs) {
    throw std::runtime_error(
      "Output tensor count mismatch: expected " +
      std::to_string(m_numOutputs) + ", got " +
      std::to_string(outputTensors.size()));
  }

  std::vector<DataOutputType> outputData;
  outputData.reserve(m_numOutputs);

  for (auto & elem : outputTensors) {
    outputData.emplace_back(
      std::make_pair(
        std::move(elem.GetTensorMutableData<float>()),
        elem.GetTensorTypeAndShapeInfo().GetShape()));
  }
  return outputData;
}

}  // namespace Ort
