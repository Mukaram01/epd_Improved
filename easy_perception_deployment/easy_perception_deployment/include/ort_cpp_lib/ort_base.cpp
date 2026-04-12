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
#include <sstream>
#include <memory>
#include <string>
#include <vector>

#include "ort_base.hpp"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

#if USE_GPU
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
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

namespace Ort
{

class OrtBase::OrtBaseImpl
{
public:
  OrtBaseImpl(
    const std::string & modelPath,         //
    const boost::optional<size_t> & gpuIdx,  //
    const boost::optional<int> & intraOpNumThreads,
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

// Constructor
OrtBase::OrtBase(
  const std::string & modelPath,
  const boost::optional<size_t> & gpuIdx,
  const boost::optional<int> & intraOpNumThreads,
  const boost::optional<std::vector<std::vector<int64_t>>> & inputShapes,
  const boost::optional<bool> & logModelInfo)
: base_impl_(
    std::make_unique<OrtBaseImpl>(
      modelPath, gpuIdx, intraOpNumThreads, inputShapes, logModelInfo))
{}

// Destructor
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

// Constructor
OrtBase::OrtBaseImpl::OrtBaseImpl(
  const std::string & modelPath,         //
  const boost::optional<size_t> & gpuIdx,  //
  const boost::optional<int> & intraOpNumThreads,
  const boost::optional<std::vector<std::vector<int64_t>>> & inputShapes,
  const boost::optional<bool> & logModelInfo)
: m_session(nullptr),
  m_env(nullptr),
  m_ortAllocator(),
  m_gpuIdx(gpuIdx),
  m_intraOpNumThreads(intraOpNumThreads),
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

  if (m_intraOpNumThreads.is_initialized()) {
    const int thread_count = m_intraOpNumThreads.value();
    if (thread_count > 0) {
      sessionOptions.SetIntraOpNumThreads(thread_count);
    }
  }

  #if USE_GPU
  if (m_gpuIdx.is_initialized()) {
    Ort::ThrowOnError(
      OrtSessionOptionsAppendExecutionProvider_CUDA(
        sessionOptions,
        m_gpuIdx.value()));
  }
  #endif

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
    // If m_inputShapes not initialized,
    // then look at m_session and derive.
    // Ensures that m_inputShapes is filled properly before use.
    // Always query TypeInfo to capture the element type declared by the model.
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

    // DEBUG
    // Identified potential failure point for loading point.
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

// Run ORT session on processed input image.
std::vector<OrtBase::DataOutputType> OrtBase::OrtBaseImpl::operator()(
  const std::vector<float *> & inputData)
{
  if (m_numInputs != inputData.size()) {
    throw std::runtime_error("Mismatch size of input data\n");
  }
  // Investigate if this statement means it is using CPU instead of GPU when GPU is intended.
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // Create inputTensors
  std::vector<Ort::Value> inputTensors;
  inputTensors.reserve(m_numInputs);
  // uint8 conversion buffers; must outlive inputTensors until Run() completes.
  std::vector<std::vector<uint8_t>> uint8Buffers;
  // Populate inputTensors with device-specific memoryInfo, the input image and the inputShapes.
  for (int i = 0; i < m_numInputs; ++i) {
    if (m_inputElementTypes[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      // The model expects raw uint8 pixel values (0-255).
      // The preprocessing path has already skipped mean subtraction and stored
      // float values in the 0-255 range; clamp and cast them here.
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
  // INFERENCE DONE HERE.
  auto outputTensors = m_session.Run(
    Ort::RunOptions{nullptr},
    m_inputNodeNames.data(),
    inputTensors.data(),
    m_numInputs,
    m_outputNodeNames.data(),
    m_numOutputs);

  // Check if outputTensors is empty. It should not be, even if it is garbage.
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
