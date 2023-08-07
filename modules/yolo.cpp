#include "yolo.h"

#include "preprocess.h"
#include "yolo_config_parser.h"

#include <omp.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

using namespace nvinfer1;
REGISTER_TENSORRT_PLUGIN(DetectPluginCreator);

Yolo::Yolo(const NetworkInfo & networkInfo, const InferParams & inferParams)
: m_NetworkType(networkInfo.networkType),
  m_ConfigFilePath(networkInfo.configFilePath),
  m_WtsFilePath(networkInfo.wtsFilePath),
  m_LabelsFilePath(networkInfo.labelsFilePath),
  m_Precision(networkInfo.precision),
  m_DeviceType(networkInfo.deviceType),
  m_CalibImages(inferParams.calibImages),
  m_CalibImagesFilePath(inferParams.calibImagesPath),
  m_CalibTableFilePath(networkInfo.calibrationTablePath),
  m_InputBlobName(networkInfo.inputBlobName),
  m_InputH(0),
  m_InputW(0),
  m_InputC(0),
  m_InputSize(0),
  m_ProbThresh(inferParams.probThresh),
  m_NMSThresh(inferParams.nmsThresh),
  m_PrintPerfInfo(inferParams.printPerfInfo),
  m_PrintPredictions(inferParams.printPredictionInfo),
  m_Logger(Logger()),
  m_Network(nullptr),
  m_Builder(nullptr),
  m_ModelStream(nullptr),
  m_Engine(nullptr),
  m_Context(nullptr),
  m_InputBindingIndex(-1),
  m_CudaStream(nullptr),
  m_model_profiler("Model"),
  m_host_profiler("Host"),
  _n_yolo_ind(0)
{
  m_configBlocks = parseConfigFile(m_ConfigFilePath);
  m_dla = networkInfo.dla;
  m_h_img = NULL;
  m_d_img = NULL;

  parseConfigBlocks();

  m_BatchSize = networkInfo.batch;
  if (networkInfo.width && networkInfo.height) {
    m_InputW = networkInfo.width;
    m_InputH = networkInfo.height;
    m_InputSize = m_InputC * m_InputH * m_InputW;
    m_EnginePath = networkInfo.data_path + "-" + std::to_string(m_InputW) + "x" +
                   std::to_string(m_InputH) + "-" + m_Precision;
    if (get_multi_precision_flg()) {
      m_EnginePath += "-multiprecision";
    }
    if (m_dla != -1) {
      m_EnginePath += "-DLA" + std::to_string(m_dla);
    }
    m_EnginePath += "-batch" + std::to_string(m_BatchSize) + ".engine";
  } else {
    m_EnginePath = networkInfo.data_path + "-" + m_Precision;
    if (get_multi_precision_flg()) {
      m_EnginePath += "-multiprecision";
    }
    if (m_dla != -1) {
      m_EnginePath += "-DLA" + std::to_string(m_dla);
    }
    m_EnginePath += "-batch" + std::to_string(m_BatchSize) + ".engine";
  }
  if (m_Precision == "kFLOAT") {
    createYOLOEngine();
  } else if (m_Precision == "kINT8") {
    Int8EntropyCalibrator calibrator(
      m_BatchSize, m_CalibImages, m_CalibImagesFilePath, m_CalibTableFilePath, m_InputSize,
      m_InputH, m_InputW, m_InputBlobName, m_NetworkType);
    createYOLOEngine(nvinfer1::DataType::kINT8, &calibrator);
  } else if (m_Precision == "kHALF") {
    createYOLOEngine(nvinfer1::DataType::kHALF, nullptr);
  } else {
    std::cout << "Unrecognized precision type " << m_Precision << std::endl;
    assert(0);
  }

  // assert(m_PluginFactory != nullptr);
  m_Engine = loadTRTEngine(m_EnginePath, /* m_PluginFactory,*/ m_Logger, m_dla);
  assert(m_Engine != nullptr);
  m_Context = m_Engine->createExecutionContext();
  assert(m_Context != nullptr);
  if (inferParams.prof) {
    m_Context->setProfiler(&m_model_profiler);
  }
  m_InputBindingIndex = m_Engine->getBindingIndex(m_InputBlobName.c_str());
  assert(m_InputBindingIndex != -1);
  assert(m_BatchSize <= static_cast<uint32_t>(m_Engine->getMaxBatchSize()));
  allocateBuffers();
  NV_CUDA_CHECK(cudaStreamCreate(&m_CudaStream));
  assert(verifyYoloEngine());
}

Yolo::~Yolo()
{
  for (auto & tensor : m_OutputTensors) NV_CUDA_CHECK(cudaFreeHost(tensor.hostBuffer));
  for (auto & deviceBuffer : m_DeviceBuffers) NV_CUDA_CHECK(cudaFree(deviceBuffer));
  NV_CUDA_CHECK(cudaStreamDestroy(m_CudaStream));
  if (m_Context) {
    m_Context->destroy();
    m_Context = nullptr;
  }

  if (m_Engine) {
    m_Engine->destroy();
    m_Engine = nullptr;
  }
}

std::vector<int> split_layer_index(const std::string & s_, const std::string & delimiter_)
{
  std::vector<int> index;
  std::string s = s_;
  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delimiter_)) != std::string::npos) {
    token = s.substr(0, pos);
    index.push_back(std::stoi(trim(token)));
    s.erase(0, pos + delimiter_.length());
  }
  index.push_back(std::stoi(trim(s)));
  return index;
}

void Yolo::createYOLOEngine(const nvinfer1::DataType dataType, Int8EntropyCalibrator * calibrator)
{
  std::vector<float> weights = loadWeights(m_WtsFilePath, m_NetworkType);
  std::vector<nvinfer1::Weights> trtWeights;
  int weightPtr = 0;
  int channels = m_InputC;
  m_Builder = nvinfer1::createInferBuilder(m_Logger);
  nvinfer1::IBuilderConfig * config = m_Builder->createBuilderConfig();

  const auto explicitBatch =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  m_Network = m_Builder->createNetworkV2(explicitBatch);
  if (
    (dataType == nvinfer1::DataType::kINT8 && !m_Builder->platformHasFastInt8()) ||
    (dataType == nvinfer1::DataType::kHALF && !m_Builder->platformHasFastFp16())) {
    std::cout << "Platform doesn't support this precision." << std::endl;
    assert(0);
  }

  nvinfer1::ITensor * data = m_Network->addInput(
    m_InputBlobName.c_str(), nvinfer1::DataType::kFLOAT,
    nvinfer1::Dims{
      4, static_cast<int>(1), static_cast<int>(m_InputC), static_cast<int>(m_InputH),
      static_cast<int>(m_InputW)});
  assert(data != nullptr);

  nvinfer1::ITensor * previous = data;
  std::vector<nvinfer1::ITensor *> tensorOutputs;
  uint32_t outputTensorCount = 0;
  uint32_t segmenter_count = 0;
  int sparse = 0;
  float total_gflops = 0.0;
  long long int total_num_params = 0;
  // build the network using the network API
  for (uint32_t i = 0; i < m_configBlocks.size(); ++i) {
    // check if num. of channels is correct
    assert(getNumChannels(previous) == channels);
    std::string layerIndex = "(" + std::to_string(i) + ")";

    if (m_configBlocks.at(i).at("type") == "net") {
      printLayerInfo("", "layer", "     inp_size", "     out_size", "weightPtr");
    } else if (m_configBlocks.at(i).at("type") == "convolutional") {
      std::string inputVol = dimsToString(previous->getDimensions());
      nvinfer1::ILayer * out;
      std::string layerType;
      // check activation
      std::string activation = "";
      float gflops = get_gflops(m_configBlocks.at(i), previous);
      int num_params = get_num_params(m_configBlocks.at(i), previous);
      if (m_configBlocks.at(i).find("sparse") != m_configBlocks.at(i).end()) {
        sparse = 1;
      }
      if (m_configBlocks.at(i).find("activation") != m_configBlocks.at(i).end()) {
        activation = m_configBlocks[i]["activation"];
      }
      // check if batch_norm enabled
      if (
        (m_configBlocks.at(i).find("batch_normalize") != m_configBlocks.at(i).end()) &&
        ("leaky" == activation)) {
        out = netAddConvBNLeaky(
          i, m_configBlocks.at(i), weights, trtWeights, weightPtr, channels, previous, m_Network);
        layerType = "conv-bn-leaky";
      } else if (
        (m_configBlocks.at(i).find("batch_normalize") != m_configBlocks.at(i).end()) &&
        ("relu" == activation)) {
        out = netAddConvBNRelu(
          i, m_configBlocks.at(i), weights, trtWeights, weightPtr, channels, previous, m_Network);
        layerType = "conv-bn-relu";

      } else if (
        (m_configBlocks.at(i).find("batch_normalize") != m_configBlocks.at(i).end()) &&
        ("swish" == activation)) {
        out = netAddConvBNSwish(
          i, m_configBlocks.at(i), weights, trtWeights, weightPtr, channels, previous, m_Network);
        layerType = "conv-bn-swish";
      } else if (
        (m_configBlocks.at(i).find("batch_normalize") != m_configBlocks.at(i).end()) &&
        ("mish" == activation)) {
        out = net_conv_bn_mish(
          i, m_configBlocks.at(i), weights, trtWeights, weightPtr, channels, previous, m_Network);
        layerType = "conv-bn-mish";
      } else {
        if ("linear" == activation) {
          out = netAddConvLinear(
            i, m_configBlocks.at(i), weights, trtWeights, weightPtr, channels, previous, m_Network);
          layerType = "conv-linear";
        } else if ("logistic" == activation) {
          out = netAddConvSigmoid(
            i, m_configBlocks.at(i), weights, trtWeights, weightPtr, channels, previous, m_Network);
          layerType = "conv-logistic4";
        } else {
          assert(1);
        }
      }
      previous = out->getOutput(0);
      assert(previous != nullptr);
      channels = getNumChannels(previous);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(out->getOutput(0));
      if (m_configBlocks.at(i).find("size") != m_configBlocks.at(i).end()) {
        int kernel = stoi(m_configBlocks.at(i).at("size"));
        layerType += " " + std::to_string(kernel) + "x" + std::to_string(kernel);
      }
      if (m_configBlocks.at(i).find("dilation") != m_configBlocks.at(i).end()) {
        int dilation = stoi(m_configBlocks.at(i).at("dilation"));
        layerType += "(" + std::to_string(dilation) + ")";
      }
      total_gflops += gflops;
      total_num_params += num_params;
      layerType += " : (" + std::to_string(gflops) + " " + std::to_string(num_params) + ")";
      printLayerInfo(layerIndex, layerType, inputVol, outputVol, std::to_string(weightPtr));
    } else if (m_configBlocks.at(i).at("type") == "shortcut") {
      assert(m_configBlocks.at(i).at("activation") == "linear");
      assert(m_configBlocks.at(i).find("from") != m_configBlocks.at(i).end());
      int from = stoi(m_configBlocks.at(i).at("from"));

      std::string inputVol = dimsToString(previous->getDimensions());
      // check if indexes are correct
      assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
      assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
      assert(i + from - 1 < i - 2);
      nvinfer1::IElementWiseLayer * ew = m_Network->addElementWise(
        *tensorOutputs[i - 2], *tensorOutputs[i + from - 1], nvinfer1::ElementWiseOperation::kSUM);
      assert(ew != nullptr);
      std::string ewLayerName = "shortcut_" + std::to_string(i);
      ew->setName(ewLayerName.c_str());
      previous = ew->getOutput(0);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(ew->getOutput(0));
      printLayerInfo(layerIndex, "skip", inputVol, outputVol, "    -");
    } else if (m_configBlocks.at(i).at("type") == "yolo") {
      std::string layerName;
      if (m_configBlocks.at(i).find("scale_x_y") != m_configBlocks.at(i).end()) {
        layerName = "ScaledYoloV4_" + std::to_string(outputTensorCount);
      } else {
        layerName = "yolo_" + std::to_string(outputTensorCount);
      }
      previous->setName(layerName.c_str());
      tensorOutputs.push_back(previous);
      m_Network->markOutput(*previous);
      TensorInfo & curYoloTensor = m_OutputTensors.at(outputTensorCount);

      if (m_configBlocks.at(i).find("colormap") != m_configBlocks.at(i).end()) {
        std::string colormapString = m_configBlocks.at(i).at("colormap");
        while (!colormapString.empty()) {
          size_t npos = colormapString.find_first_of(',');
          if (npos != std::string::npos) {
            uint32_t colormap = (uint32_t)std::stoi(trim(colormapString.substr(0, npos)));
            curYoloTensor.colormap.push_back(colormap);
            colormapString.erase(0, npos + 1);
          } else {
            int colormap = (uint32_t)std::stoi(trim(colormapString));
            curYoloTensor.colormap.push_back(colormap);
            break;
          }
        }
      }

      if (m_configBlocks.at(i).find("names") != m_configBlocks.at(i).end()) {
        std::string namesString = m_configBlocks.at(i).at("names");
        while (!namesString.empty()) {
          size_t npos = namesString.find_first_of(',');
          if (npos != std::string::npos) {
            std::string name = trim(namesString.substr(0, npos));
            curYoloTensor.names.push_back(name);
            namesString.erase(0, npos + 1);
          } else {
            std::string name = trim(namesString);
            curYoloTensor.names.push_back(name);
            break;
          }
        }
      }

      ++outputTensorCount;
    } else if (m_configBlocks.at(i).at("type") == "softmax") {
      std::string inputVol = dimsToString(previous->getDimensions());
      nvinfer1::Dims prevTensorDims = previous->getDimensions();
      assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
      auto * softmax = m_Network->addSoftMax(*tensorOutputs[i - 2]);
      assert(softmax != nullptr);
      std::string softmaxLayerName = "segmenter_" + std::to_string(segmenter_count);
      softmax->setName(softmaxLayerName.c_str());
      previous = softmax->getOutput(0);

      previous->setName(softmaxLayerName.c_str());
      assert(previous != nullptr);
      m_Network->markOutput(*previous);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(softmax->getOutput(0));
      // tensorOutputs.push_back(previous);
      printLayerInfo(layerIndex, "segmenter", inputVol, outputVol, "    -");

      TensorInfo & curYoloTensor = m_OutputTensors.at(outputTensorCount);
      curYoloTensor.gridSize = prevTensorDims.d[2];
      curYoloTensor.grid_h = prevTensorDims.d[2];
      curYoloTensor.grid_w = prevTensorDims.d[3];
      curYoloTensor.numClasses = prevTensorDims.d[1];
      curYoloTensor.blobName = softmaxLayerName;
      curYoloTensor.volume = curYoloTensor.grid_h * curYoloTensor.grid_w * curYoloTensor.numClasses;
      ++outputTensorCount;

      if (m_configBlocks.at(i).find("colormap") != m_configBlocks.at(i).end()) {
        std::string colormapString = m_configBlocks.at(i).at("colormap");
        while (!colormapString.empty()) {
          size_t npos = colormapString.find_first_of(',');
          if (npos != std::string::npos) {
            uint32_t colormap = std::stoi(trim(colormapString.substr(0, npos)));
            curYoloTensor.colormap.push_back(colormap);
            colormapString.erase(0, npos + 1);
          } else {
            int colormap = std::stoi(trim(colormapString));
            curYoloTensor.colormap.push_back(colormap);
            break;
          }
        }
      }
      curYoloTensor.depth = false;
      if (m_configBlocks.at(i).find("depth") != m_configBlocks.at(i).end()) {
        curYoloTensor.depth = true;
      }

      ++segmenter_count;
    } else if (m_configBlocks.at(i).at("type") == "route") {
      size_t found = m_configBlocks.at(i).at("layers").find(",");
      if (found != std::string::npos)  // concate multi layers
      {
        std::vector<int> vec_index = split_layer_index(m_configBlocks.at(i).at("layers"), ",");
        for (auto & ind_layer : vec_index) {
          if (ind_layer < 0) {
            ind_layer = static_cast<int>(tensorOutputs.size()) + ind_layer;
          }
          assert(ind_layer < static_cast<int>(tensorOutputs.size()) && ind_layer >= 0);
        }
        nvinfer1::ITensor ** concatInputs = reinterpret_cast<nvinfer1::ITensor **>(
          malloc(sizeof(nvinfer1::ITensor *) * vec_index.size()));
        for (size_t ind = 0; ind < vec_index.size(); ++ind) {
          concatInputs[ind] = tensorOutputs[vec_index[ind]];
        }
        nvinfer1::IConcatenationLayer * concat =
          m_Network->addConcatenation(concatInputs, static_cast<int>(vec_index.size()));
        assert(concat != nullptr);
        std::string concatLayerName = "route_" + std::to_string(i - 1);
        concat->setName(concatLayerName.c_str());
        // concatenate along the channel dimension
        concat->setAxis(1);
        previous = concat->getOutput(0);
        assert(previous != nullptr);
        nvinfer1::Dims debug = previous->getDimensions();
        std::string outputVol = dimsToString(previous->getDimensions());
        int nums = 0;
        for (auto & indx : vec_index) {
          nums += getNumChannels(tensorOutputs[indx]);
        }
        channels = nums;
        tensorOutputs.push_back(concat->getOutput(0));
        printLayerInfo(layerIndex, "route", "        -", outputVol, std::to_string(weightPtr));
      } else  // single layer
      {
        int idx = std::stoi(trim(m_configBlocks.at(i).at("layers")));
        if (idx < 0) {
          idx = static_cast<int>(tensorOutputs.size()) + idx;
        }
        assert(idx < static_cast<int>(tensorOutputs.size()) && idx >= 0);

        // route
        if (m_configBlocks.at(i).find("groups") == m_configBlocks.at(i).end()) {
          previous = tensorOutputs[idx];
          assert(previous != nullptr);
          std::string outputVol = dimsToString(previous->getDimensions());
          // set the output volume depth
          channels = getNumChannels(tensorOutputs[idx]);
          tensorOutputs.push_back(tensorOutputs[idx]);
          printLayerInfo(layerIndex, "route", "        -", outputVol, std::to_string(weightPtr));

        }
        // yolov4-tiny route split layer
        else {
          if (m_configBlocks.at(i).find("group_id") == m_configBlocks.at(i).end()) {
            assert(0);
          }
          auto dim = tensorOutputs[idx]->getDimensions();
          int groups = stoi(m_configBlocks.at(i).at("groups"));
          auto start = nvinfer1::Dims{dim.nbDims, {0, dim.d[1] / groups, 0, 0}};
          // auto start = nvinfer1::Dims{dim.nbDims, {0, 0, 0}};
          auto size = nvinfer1::Dims{dim.nbDims, {1, dim.d[1] / groups, dim.d[2], dim.d[3]}};
          auto stride = nvinfer1::Dims{4, 1, 1, 1, 1};
          nvinfer1::ISliceLayer * out =
            m_Network->addSlice(*(tensorOutputs[idx]), start, size, stride);
          std::string inputVol = dimsToString(previous->getDimensions());
          std::string sliceLayerName = "slice_" + std::to_string(i - 1);
          out->setName(sliceLayerName.c_str());
          previous = out->getOutput(0);
          assert(previous != nullptr);
          channels = getNumChannels(previous);
          std::string outputVol = dimsToString(previous->getDimensions());
          tensorOutputs.push_back(out->getOutput(0));
          printLayerInfo(layerIndex, "slice", inputVol, outputVol, std::to_string(weightPtr));
        }
      }
    } else if (m_configBlocks.at(i).at("type") == "upsample2") {
      std::string inputVol = dimsToString(previous->getDimensions());
      nvinfer1::ILayer * out = netAddUpsample(
        i - 1, m_configBlocks[i], weights, trtWeights, channels, previous, m_Network);
      previous = out->getOutput(0);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(out->getOutput(0));
      printLayerInfo(layerIndex, "upsample", inputVol, outputVol, "    -");
    } else if (m_configBlocks.at(i).at("type") == "upsample") {
      std::string inputVol = dimsToString(previous->getDimensions());
      nvinfer1::IResizeLayer * resize = m_Network->addResize(*previous);
      // std::vector<float> scales{1,1,2,2};
      // resize->setScales(scales.data(), scales.size());
      // resize->setResizeMode(nvinfer1::InterpolationMode::kNEAREST);
      resize->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
      nvinfer1::Dims d = nvinfer1::Dims{
        4, 1, previous->getDimensions().d[1], previous->getDimensions().d[2] * 2,
        previous->getDimensions().d[3] * 2};
      resize->setOutputDimensions(d);
      previous = resize->getOutput(0);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(resize->getOutput(0));
      printLayerInfo(layerIndex, "resize", inputVol, outputVol, "    -");
    } else if (m_configBlocks.at(i).at("type") == "maxpool") {
      // Add same padding layers
      float gflops = get_gflops(m_configBlocks.at(i), previous);
      if (m_configBlocks.at(i).at("size") == "2" && m_configBlocks.at(i).at("stride") == "1") {
        //  m_TinyMaxpoolPaddingFormula->addSamePaddingLayer("maxpool_" + std::to_string(i));
      }
      std::string inputVol = dimsToString(previous->getDimensions());
      nvinfer1::ILayer * out = netAddMaxpool(i, m_configBlocks.at(i), previous, m_Network);
      previous = out->getOutput(0);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(out->getOutput(0));
      total_gflops += gflops;
      std::string layerType = "maxpool";
      layerType += " : (" + std::to_string(gflops) + ")";
      printLayerInfo(layerIndex, layerType, inputVol, outputVol, std::to_string(weightPtr));
    } else {
      std::cout << "Unsupported layer type --> \"" << m_configBlocks.at(i).at("type") << "\""
                << std::endl;
      assert(0);
    }
  }
  std::cout << "Total GFLOPS " << std::to_string(total_gflops) << " Total MACC "
            << std::to_string(total_gflops / 2) << std::endl;
  std::cout << "Total Params[M] " << std::to_string(total_num_params / (float)(1000 * 1000))
            << std::endl;

  int num = m_Network->getNbLayers();
  for (int i = 0; i < num; i++) {
    nvinfer1::ILayer * layer = m_Network->getLayer(i);
    auto ltype = layer->getType();
    std::string name = layer->getName();
    m_model_profiler.setProfDict(layer);
  }
  if (static_cast<int>(weights.size()) != weightPtr) {
    std::cout << "Number of unused weights left : " << static_cast<int>(weights.size()) - weightPtr
              << std::endl;
    assert(0);
  }

  // Create and cache the engine if not already present
  if (fileExists(m_EnginePath)) {
    std::cout << "Using previously generated plan file located at " << m_EnginePath << std::endl;
    destroyNetworkUtils(trtWeights);
    return;
  }

  m_Builder->setMaxBatchSize(m_BatchSize);

  config->setMaxWorkspaceSize(1 << 30);
  //    std::cout << "kENABLE_TACTIC_HEURISTIC" << std::endl;
  // config->setFlag(nvinfer1::BuilderFlag::kENABLE_TACTIC_HEURISTIC);
  if (sparse) {
    config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
    std::cout << "Set kSPARSE WEIGHTS" << std::endl;
  }
  if (get_multi_precision_flg()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  if (m_dla != -1) {
    int ndla = m_Builder->getNbDLACores();
    if (ndla > 0) {
      std::cout << "###" << ndla << " DLAs are supported! ###" << std::endl;
    } else {
      std::cout << "###Warninig : "
                << "No DLA is supported! ###" << std::endl;
    }
    config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    config->setDLACore(m_dla);
#if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8200
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
#else
    config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
#endif
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
  }
  if (dataType == nvinfer1::DataType::kINT8) {
    assert((calibrator != nullptr) && "Invalid calibrator for INT8 precision");

    config->setFlag(nvinfer1::BuilderFlag::kINT8);
#if (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSOR_PATCH >= 8200
    config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
#endif
    config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    config->setInt8Calibrator(calibrator);
  } else if (dataType == nvinfer1::DataType::kHALF) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  // Build the engine
  std::cout << "Building the TensorRT Engine..." << std::endl;
  m_Engine = m_Builder->buildEngineWithConfig(*m_Network, *config);
  assert(m_Engine != nullptr);
  std::cout << "Building complete!" << std::endl;

  // Serialize the engine
  writePlanFileToDisk();

  // destroy
  destroyNetworkUtils(trtWeights);
}

void Yolo::preprocess_gpu(unsigned char * src, int w, int h)
{
  float norm = 1 / 255.0;
  int src_size = 3 * w * h;
  if (!m_h_img) {
    NV_CUDA_CHECK(cudaMallocHost((void **)&m_h_img, sizeof(unsigned char) * src_size));
    NV_CUDA_CHECK(cudaMalloc((void **)&m_d_img, sizeof(unsigned char) * src_size));
  }
  // Copy into pinned memory
  memcpy(m_h_img, src, src_size * sizeof(unsigned char));
  // Copy into device memory
  NV_CUDA_CHECK(cudaMemcpyAsync(
    m_d_img, m_h_img, src_size * sizeof(unsigned char), cudaMemcpyHostToDevice, m_CudaStream));

  blobFromImageGpu(
    (float *)(m_DeviceBuffers.at(m_InputBindingIndex)), m_d_img, m_InputW, m_InputH, m_InputC, w, h,
    3, norm, m_CudaStream);
}

void Yolo::doInference(const unsigned char * input, const uint32_t batchSize)
{
  Timer timer;
  assert(batchSize <= m_BatchSize && "Image batch size exceeds TRT engines batch size");
  if (input) {
    NV_CUDA_CHECK(cudaMemcpyAsync(
      m_DeviceBuffers.at(m_InputBindingIndex), input, batchSize * m_InputSize * sizeof(float),
      cudaMemcpyHostToDevice, m_CudaStream));
  }

  // m_Context->enqueue(batchSize, m_DeviceBuffers.data(), m_CudaStream, nullptr);
  m_Context->enqueueV2(m_DeviceBuffers.data(), m_CudaStream, nullptr);

  for (auto & tensor : m_OutputTensors) {
    NV_CUDA_CHECK(cudaMemcpyAsync(
      tensor.hostBuffer, m_DeviceBuffers.at(tensor.bindingIndex),
      batchSize * tensor.volume * sizeof(float), cudaMemcpyDeviceToHost, m_CudaStream));
  }
  cudaStreamSynchronize(m_CudaStream);
  timer.out("inference");
}

void Yolo::doProfiling(const unsigned char * input, const uint32_t batchSize)
{
  Timer timer;
  assert(batchSize <= m_BatchSize && "Image batch size exceeds TRT engines batch size");
  if (input) {
    NV_CUDA_CHECK(cudaMemcpyAsync(
      m_DeviceBuffers.at(m_InputBindingIndex), input, batchSize * m_InputSize * sizeof(float),
      cudaMemcpyHostToDevice, m_CudaStream));
  }
  std::chrono::high_resolution_clock::time_point start;
  start = std::chrono::high_resolution_clock::now();

  // m_Context->enqueue(batchSize, m_DeviceBuffers.data(), m_CudaStream, nullptr);
  m_Context->enqueueV2(m_DeviceBuffers.data(), m_CudaStream, nullptr);
  // cudaStreamSynchronize(m_CudaStream);
  auto now = std::chrono::high_resolution_clock::now();
  m_host_profiler.reportLayerTime(
    "inference", std::chrono::duration<float, std::milli>(now - start).count());

  for (auto & tensor : m_OutputTensors) {
    NV_CUDA_CHECK(cudaMemcpyAsync(
      tensor.hostBuffer, m_DeviceBuffers.at(tensor.bindingIndex),
      batchSize * tensor.volume * sizeof(float), cudaMemcpyDeviceToHost, m_CudaStream));
  }
  cudaStreamSynchronize(m_CudaStream);
  timer.out("inference");
}

std::vector<BBoxInfo> Yolo::decodeDetections(
  const int & imageIdx, const int & imageH, const int & imageW)
{
  std::vector<BBoxInfo> binfo;
  for (auto & tensor : m_OutputTensors) {
    if (!tensor.segmenter) {
      std::vector<BBoxInfo> curBInfo = decodeTensor(imageIdx, imageH, imageW, tensor);
      binfo.insert(binfo.end(), curBInfo.begin(), curBInfo.end());
    }
  }

  return binfo;
}

uint32_t * Yolo::get_detection_colormap(void)
{
  uint32_t * ret = NULL;
  for (auto & tensor : m_OutputTensors) {
    if (!tensor.segmenter) {
      if (tensor.colormap.size()) {
        ret = tensor.colormap.data();
      }
    }
  }
  return ret;
}

std::vector<std::string> Yolo::get_detection_names(int id)
{
  std::vector<std::string> names;
  int i;
  for (i = 0; i < (int)m_OutputTensors.size(); i++) {
    if (i == id) {
      names = m_OutputTensors[i].names;
    }
  }
  return names;
}

std::vector<cv::Mat> Yolo::apply_argmax(const int & imageIdx)
{
  std::vector<cv::Mat> segmentation;
  for (auto & tensor : m_OutputTensors) {
    if (tensor.segmenter) {
      const float * prob = &tensor.hostBuffer[imageIdx * tensor.volume];
      cv::Mat argmax = cv::Mat::zeros(tensor.grid_h, tensor.grid_w, CV_8UC1);
      int cstart = tensor.depth == true ? 1 : 0;
      // #pragma omp parallel for
      for (int y = 0; y < tensor.grid_h; y++) {
        for (int x = 0; x < tensor.grid_w; x++) {
          float max = 0.0;
          int index = 0;
          for (int c = cstart; c < tensor.numClasses; c++) {
            float value = prob[c * tensor.grid_h * tensor.grid_w + y * tensor.grid_w + x];
            if (max < value) {
              max = value;
              index = c;
            }
          }
          argmax.at<unsigned char>(y, x) = index;
        }
      }
      segmentation.push_back(argmax);
    }
  }
  return segmentation;
}

std::vector<cv::Mat> Yolo::get_colorlbl(std::vector<cv::Mat> & argmax)
{
  std::vector<cv::Mat> segmentation;
  int count = 0;
  for (auto & tensor : m_OutputTensors) {
    if (tensor.segmenter && !tensor.depth) {
      cv::Mat gray = argmax[count];
      cv::Mat mask = cv::Mat::zeros(tensor.grid_h, tensor.grid_w, CV_8UC3);
      for (int y = 0; y < tensor.grid_h; y++) {
        for (int x = 0; x < tensor.grid_w; x++) {
          int id = gray.at<unsigned char>(y, x);
          std::vector<unsigned int> colormap = tensor.colormap;
          mask.at<cv::Vec3b>(y, x)[0] = colormap[3 * id + 2];
          mask.at<cv::Vec3b>(y, x)[1] = colormap[3 * id + 1];
          mask.at<cv::Vec3b>(y, x)[2] = colormap[3 * id + 0];
        }
      }
      segmentation.push_back(mask);
    }
    if (tensor.segmenter) {
      count++;
    }
  }
  return segmentation;
}

std::vector<cv::Mat> Yolo::get_depthmap(std::vector<cv::Mat> & argmax)
{
  std::vector<cv::Mat> segmentation;
  int count = 0;

  for (auto & tensor : m_OutputTensors) {
    if (tensor.segmenter && tensor.depth) {
      cv::Mat gray = argmax[count];
      /*
      cv::Mat mask = cv::Mat::zeros(tensor.grid_h, tensor.grid_w, CV_8UC3);      int c =
      tensor.numClasses; std::cout << c << std::endl; for (int y = 0; y < tensor.grid_h; y++) { for
      (int x = 0; x < tensor.grid_w; x++) { int id = gray.at<unsigned char>(y, x);
          mask.at<cv::Vec3b>(y, x)[0] = (unsigned char)(255 * id/(float)c);
          mask.at<cv::Vec3b>(y, x)[1] = (unsigned char)(255 * id/(float)c);
          mask.at<cv::Vec3b>(y, x)[2] = (unsigned char)(255 * id/(float)c);
        }
      }
      */
      cv::Mat hsv = cv::Mat::zeros(tensor.grid_h, tensor.grid_w, CV_8UC3);
      int c = tensor.numClasses;
      for (int y = 0; y < tensor.grid_h; y++) {
        for (int x = 0; x < tensor.grid_w; x++) {
          int id = gray.at<unsigned char>(y, x);
          float rel = id / (float)c;
          // int tmp = 120 + 90 * (1.0-rel);
          // tmp = tmp > 60.0 ? 60.0 : tmp;
          int hue = 120 + 90 * (1.0 - rel);
          hue = hue > 180 ? (180 - hue) * (-1) : hue;
          unsigned char val = 255 - 180 * (1.0 - rel);
          // hue = hue < 300 ? 0 : hue;
          hsv.at<cv::Vec3b>(y, x)[0] = hue;
          hsv.at<cv::Vec3b>(y, x)[1] = 255;
          hsv.at<cv::Vec3b>(y, x)[2] = val;
        }
      }
      cv::Mat mask;
      cv::cvtColor(hsv, mask, cv::COLOR_HSV2RGB);
      segmentation.push_back(mask);
    }
    if (tensor.segmenter) {
      count++;
    }
  }
  return segmentation;
}

std::vector<std::map<std::string, std::string>> Yolo::parseConfigFile(const std::string cfgFilePath)
{
  assert(fileExists(cfgFilePath));
  std::ifstream file(cfgFilePath);
  assert(file.good());
  std::cout << "Parse from ... " << cfgFilePath << std::endl;
  std::string line;
  std::vector<std::map<std::string, std::string>> blocks;
  std::map<std::string, std::string> block;

  while (getline(file, line)) {
    line = trim(line);
    if (line.empty()) continue;
    if (line.front() == '#') continue;
    if (line.front() == '[') {
      if (!block.empty()) {
        blocks.push_back(block);
        block.clear();
      }
      std::string key = "type";
      std::string value = trim(line.substr(1, line.size() - 2));
      block.insert(std::pair<std::string, std::string>(key, value));
    } else {
      size_t cpos = line.find('=');
      std::string key = trim(line.substr(0, cpos));
      std::string value = trim(line.substr(cpos + 1));
      block.insert(std::pair<std::string, std::string>(key, value));
    }
  }
  blocks.push_back(block);
  return blocks;
}

void Yolo::parseConfigBlocks()
{
  int segmenter_count = 0;
  for (auto block : m_configBlocks) {
    if (block.at("type") == "net") {
      assert((block.find("height") != block.end()) && "Missing 'height' param in network cfg");
      assert((block.find("width") != block.end()) && "Missing 'width' param in network cfg");
      assert((block.find("channels") != block.end()) && "Missing 'channels' param in network cfg");
      assert((block.find("batch") != block.end()) && "Missing 'batch' param in network cfg");

      m_InputH = std::stoul(trim(block.at("height")));
      m_InputW = std::stoul(trim(block.at("width")));
      m_InputC = std::stoul(trim(block.at("channels")));
      m_BatchSize = std::stoi(trim(block.at("batch")));
      //   assert(m_InputW == m_InputH);
      m_InputSize = m_InputC * m_InputH * m_InputW;
    } else if ((block.at("type") == "region") || (block.at("type") == "yolo")) {
      assert(
        (block.find("num") != block.end()) &&
        std::string("Missing 'num' param in " + block.at("type") + " layer").c_str());
      assert(
        (block.find("classes") != block.end()) &&
        std::string("Missing 'classes' param in " + block.at("type") + " layer").c_str());
      assert(
        (block.find("anchors") != block.end()) &&
        std::string("Missing 'anchors' param in " + block.at("type") + " layer").c_str());

      TensorInfo outputTensor;
      std::string anchorString = block.at("anchors");
      while (!anchorString.empty()) {
        size_t npos = anchorString.find_first_of(',');
        if (npos != std::string::npos) {
          float anchor = std::stof(trim(anchorString.substr(0, npos)));
          outputTensor.anchors.push_back(anchor);
          anchorString.erase(0, npos + 1);
        } else {
          float anchor = std::stof(trim(anchorString));
          outputTensor.anchors.push_back(anchor);
          break;
        }
      }

      if (
        (m_NetworkType == "yolov3") || (m_NetworkType == "yolov3-tiny") ||
        (m_NetworkType == "yolov4") || (m_NetworkType == "yolov4-tiny")) {
        assert(
          (block.find("mask") != block.end()) &&
          std::string("Missing 'mask' param in " + block.at("type") + " layer").c_str());

        std::string maskString = block.at("mask");
        while (!maskString.empty()) {
          size_t npos = maskString.find_first_of(',');
          if (npos != std::string::npos) {
            uint32_t mask = std::stoul(trim(maskString.substr(0, npos)));
            outputTensor.masks.push_back(mask);
            maskString.erase(0, npos + 1);
          } else {
            uint32_t mask = std::stoul(trim(maskString));
            outputTensor.masks.push_back(mask);
            break;
          }
        }
      }

      outputTensor.numBBoxes = outputTensor.masks.size() > 0 ? outputTensor.masks.size()
                                                             : std::stoul(trim(block.at("num")));
      outputTensor.numClasses = std::stoul(block.at("classes"));
      if (m_ClassNames.empty()) {
        for (uint32_t i = 0; i < outputTensor.numClasses; ++i) {
          m_ClassNames.push_back(std::to_string(i));
        }
      }
      if (block.find("new_coords") != block.end()) {
        outputTensor.blobName = "ScaledYoloV4_" + std::to_string(_n_yolo_ind);
      } else {
        outputTensor.blobName = "yolo_" + std::to_string(_n_yolo_ind);
      }
      outputTensor.gridSize = (m_InputH / 32) * pow(2, _n_yolo_ind);
      outputTensor.grid_h = (m_InputH / 32) * pow(2, _n_yolo_ind);
      outputTensor.grid_w = (m_InputW / 32) * pow(2, _n_yolo_ind);
      if (m_NetworkType == "yolov4") {
        outputTensor.gridSize = (m_InputH / 32) * pow(2, 2 - _n_yolo_ind);
        outputTensor.grid_h = (m_InputH / 32) * pow(2, 2 - _n_yolo_ind);
        outputTensor.grid_w = (m_InputW / 32) * pow(2, 2 - _n_yolo_ind);
      }
      outputTensor.stride = m_InputH / outputTensor.gridSize;
      outputTensor.stride_h = m_InputH / outputTensor.grid_h;
      outputTensor.stride_w = m_InputW / outputTensor.grid_w;
      outputTensor.volume = outputTensor.grid_h * outputTensor.grid_w *
                            (outputTensor.numBBoxes * (5 + outputTensor.numClasses));
      m_OutputTensors.push_back(outputTensor);
      _n_yolo_ind++;
    } else if (block.at("type") == "softmax") {
      TensorInfo outputTensor;
      outputTensor.blobName = "segmenter_" + std::to_string(segmenter_count);
      outputTensor.segmenter = true;
      m_OutputTensors.push_back(outputTensor);
      segmenter_count++;
    }
  }
}

void Yolo::allocateBuffers()
{
  m_DeviceBuffers.resize(m_Engine->getNbBindings(), nullptr);
  assert(m_InputBindingIndex != -1 && "Invalid input binding index");
  NV_CUDA_CHECK(cudaMalloc(
    &m_DeviceBuffers.at(m_InputBindingIndex), m_BatchSize * m_InputSize * sizeof(float)));

  for (auto & tensor : m_OutputTensors) {
    tensor.bindingIndex = m_Engine->getBindingIndex(tensor.blobName.c_str());
    assert((tensor.bindingIndex != -1) && "Invalid output binding index");
    if (tensor.segmenter) {
      NV_CUDA_CHECK(cudaMalloc(
        &m_DeviceBuffers.at(tensor.bindingIndex), m_BatchSize * tensor.volume * sizeof(float)));
      NV_CUDA_CHECK(
        cudaMallocHost(&tensor.hostBuffer, tensor.volume * m_BatchSize * sizeof(float)));
    } else {
      NV_CUDA_CHECK(cudaMalloc(
        &m_DeviceBuffers.at(tensor.bindingIndex), m_BatchSize * tensor.volume * sizeof(float)));
      NV_CUDA_CHECK(
        cudaMallocHost(&tensor.hostBuffer, tensor.volume * m_BatchSize * sizeof(float)));
    }
  }
}

bool Yolo::verifyYoloEngine()
{
  assert(
    (m_Engine->getNbBindings() == (1 + m_OutputTensors.size()) &&
     "Binding info doesn't match between cfg and engine file \n"));

  for (auto tensor : m_OutputTensors) {
    assert(
      !strcmp(m_Engine->getBindingName(tensor.bindingIndex), tensor.blobName.c_str()) &&
      "Blobs names dont match between cfg and engine file \n");
    assert(
      get4DTensorVolume(m_Engine->getBindingDimensions(tensor.bindingIndex)) == tensor.volume &&
      "Tensor volumes dont match between cfg and engine file \n");
  }

  assert(m_Engine->bindingIsInput(m_InputBindingIndex) && "Incorrect input binding index \n");
  assert(
    m_Engine->getBindingName(m_InputBindingIndex) == m_InputBlobName &&
    "Input blob name doesn't match between config and engine file");
  assert(get4DTensorVolume(m_Engine->getBindingDimensions(m_InputBindingIndex)) == m_InputSize);
  return true;
}

void Yolo::destroyNetworkUtils(std::vector<nvinfer1::Weights> & trtWeights)
{
  if (m_Network) m_Network->destroy();
  if (m_Engine) m_Engine->destroy();
  if (m_Builder) m_Builder->destroy();
  if (m_ModelStream) m_ModelStream->destroy();

  // deallocate the weights
  for (auto & trtWeight : trtWeights) {
    if (trtWeight.count > 0) free(const_cast<void *>(trtWeight.values));
  }
}

void Yolo::writePlanFileToDisk()
{
  std::cout << "Serializing the TensorRT Engine..." << std::endl;
  assert(m_Engine && "Invalid TensorRT Engine");
  m_ModelStream = m_Engine->serialize();
  assert(m_ModelStream && "Unable to serialize engine");
  assert(!m_EnginePath.empty() && "Enginepath is empty");

  // write data to output file
  std::stringstream gieModelStream;
  gieModelStream.seekg(0, gieModelStream.beg);
  gieModelStream.write(static_cast<const char *>(m_ModelStream->data()), m_ModelStream->size());
  std::ofstream outFile;
  outFile.open(m_EnginePath, std::ios::binary | std::ios::out);
  outFile << gieModelStream.rdbuf();
  outFile.close();

  std::cout << "Serialized plan file cached at location : " << m_EnginePath << std::endl;
}

void Yolo::print_profiling()
{
  std::cout << "##Dump Profiler" << std::endl;
  std::cout << m_host_profiler;
  std::cout << std::endl;
  std::cout << m_model_profiler;
}
