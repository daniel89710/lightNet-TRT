

#include "plugin_factory.h"

#include "trt_utils.h"

/******* Yolo Layer V3 *******/
/*****************************/
namespace nvinfer1
{
YoloLayer::YoloLayer()
{
}

YoloLayer::YoloLayer(const void * data, size_t length)
{
  const char *d = static_cast<const char *>(data), *a = d;
  (void)a;  // Added by Koji Minoda
  re(d, m_NumBoxes);
  re(d, m_NumClasses);
  re(d, _n_grid_h);
  re(d, _n_grid_w);
  re(d, m_OutputSize);
  re(d, m_new_coords);
  assert(d = a + length);
}
void YoloLayer::serialize(void * buffer) const noexcept
{
  char *d = static_cast<char *>(buffer), *a = d;
  (void)a;  // Added by Koji Minoda
  wr(d, m_NumBoxes);
  wr(d, m_NumClasses);
  wr(d, _n_grid_h);
  wr(d, _n_grid_w);
  wr(d, m_OutputSize);
  wr(d, m_new_coords);
  assert(d == a + getSerializationSize());
  printf("Serialize V3\n");
}

bool YoloLayer::supportsFormat(DataType type, PluginFormat format) const noexcept
{
  return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

void YoloLayer::configureWithFormat(
  const Dims * inputDims, int nbInputs, const Dims * outputDims, int nbOutputs, DataType type,
  PluginFormat format, int maxBatchSize) noexcept
{
}

IPluginV2 * YoloLayer::clone() const noexcept
{
  YoloLayer * p = new YoloLayer(m_NumBoxes, m_NumClasses, _n_grid_h, _n_grid_w, m_new_coords);
  p->setPluginNamespace(_s_plugin_namespace.c_str());
  return p;
}

YoloLayer::YoloLayer(
  const uint32_t & numBoxes, const uint32_t & numClasses, const uint32_t & grid_h_,
  const uint32_t & grid_w_, const uint32_t & new_coords)
: m_NumBoxes(numBoxes),
  m_NumClasses(numClasses),
  _n_grid_h(grid_h_),
  _n_grid_w(grid_w_),
  m_new_coords(new_coords)
{
  assert(m_NumBoxes > 0);
  assert(m_NumClasses > 0);
  assert(_n_grid_h > 0);
  assert(_n_grid_w > 0);
  m_OutputSize = _n_grid_h * _n_grid_w * (m_NumBoxes * (4 + 1 + m_NumClasses));
}

int YoloLayer::getNbOutputs() const noexcept
{
  return 1;
}

nvinfer1::Dims YoloLayer::getOutputDimensions(
  int index, const nvinfer1::Dims * inputs, int nbInputDims) noexcept
{
  assert(index == 0);
  assert(nbInputDims == 1);
  return inputs[0];
}

// void YoloLayerV3::configure(const nvinfer1::Dims* inputDims, int nbInputs,
//                             const nvinfer1::Dims* outputDims, int nbOutputs, int maxBatchSize)
//                             noexcept
//{
//     assert(nbInputs == 1);
//     assert(inputDims != nullptr);
// }

int YoloLayer::initialize() noexcept
{
  return 0;
}

void YoloLayer::terminate() noexcept
{
}

size_t YoloLayer::getWorkspaceSize(int maxBatchSize) const noexcept
{
  return 0;
}

int YoloLayer::enqueue(
  int batchSize, const void * const * inputs, void * const * outputs, void * workspace,
  cudaStream_t stream) noexcept
{
  NV_CUDA_CHECK(cudaYoloLayerV3(
    inputs[0], outputs[0], batchSize, _n_grid_h, _n_grid_w, m_NumClasses, m_NumBoxes, m_OutputSize,
    m_new_coords, stream));
  return 0;
}

size_t YoloLayer::getSerializationSize() const noexcept
{
  return sizeof(m_NumBoxes) + sizeof(m_NumClasses) + sizeof(_n_grid_w) + sizeof(_n_grid_h) +
         sizeof(m_OutputSize) + sizeof(m_new_coords);
}

PluginFieldCollection YoloLayerPluginCreator::mFC{};
std::vector<PluginField> YoloLayerPluginCreator::mPluginAttributes;

YoloLayerPluginCreator::YoloLayerPluginCreator()
{
  mPluginAttributes.clear();

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char * YoloLayerPluginCreator::getPluginName() const noexcept
{
  return "YOLO_TRT";
}

const char * YoloLayerPluginCreator::getPluginVersion() const noexcept
{
  return "1.0";
}

const PluginFieldCollection * YoloLayerPluginCreator::getFieldNames() noexcept
{
  return &mFC;
}

IPluginV2 * YoloLayerPluginCreator::createPlugin(
  const char * name, const PluginFieldCollection * fc) noexcept
{
  YoloLayer * obj = new YoloLayer();
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

IPluginV2 * YoloLayerPluginCreator::deserializePlugin(
  const char * name, const void * serialData, size_t serialLength) noexcept
{
  // This object will be deleted when the network is destroyed, which will
  // call MishPlugin::destroy()
  YoloLayer * obj = new YoloLayer(serialData, serialLength);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

void YoloLayerPluginCreator::setPluginNamespace(const char * libNamespace) noexcept
{
  mNamespace = libNamespace;
}

const char * YoloLayerPluginCreator::getPluginNamespace() const noexcept
{
  return mNamespace.c_str();
}
}  // namespace nvinfer1

namespace nvinfer1
{
// Scaled YOLOv4
YoloV4Layer::YoloV4Layer()
{
}

YoloV4Layer::YoloV4Layer(const void * data, size_t length)
{
  const char *d = static_cast<const char *>(data), *a = d;
  (void)a;  // Added by Koji Minoda
  re(d, m_NumBoxes);
  re(d, m_NumClasses);
  re(d, _n_grid_h);
  re(d, _n_grid_w);
  re(d, m_OutputSize);
  re(d, m_new_coords);
  re(d, m_scale_x_y);
  assert(d = a + length);
}
void YoloV4Layer::serialize(void * buffer) const noexcept
{
  char *d = static_cast<char *>(buffer), *a = d;
  (void)a;  // Added by Koji Minoda
  wr(d, m_NumBoxes);
  wr(d, m_NumClasses);
  wr(d, _n_grid_h);
  wr(d, _n_grid_w);
  wr(d, m_OutputSize);
  wr(d, m_new_coords);
  wr(d, m_scale_x_y);
  assert(d == a + getSerializationSize());
}

bool YoloV4Layer::supportsFormat(DataType type, PluginFormat format) const noexcept
{
  return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
}

void YoloV4Layer::configureWithFormat(
  const Dims * inputDims, int nbInputs, const Dims * outputDims, int nbOutputs, DataType type,
  PluginFormat format, int maxBatchSize) noexcept
{
}

IPluginV2 * YoloV4Layer::clone() const noexcept
{
  YoloV4Layer * p =
    new YoloV4Layer(m_NumBoxes, m_NumClasses, _n_grid_h, _n_grid_w, m_new_coords, m_scale_x_y);
  p->setPluginNamespace(_s_plugin_namespace.c_str());
  return p;
}

YoloV4Layer::YoloV4Layer(
  const uint32_t & numBoxes, const uint32_t & numClasses, const uint32_t & grid_h_,
  const uint32_t & grid_w_, const uint32_t & new_coords, const float & scale_x_y)
: m_NumBoxes(numBoxes),
  m_NumClasses(numClasses),
  _n_grid_h(grid_h_),
  _n_grid_w(grid_w_),
  m_scale_x_y(scale_x_y),
  m_new_coords(new_coords)
{
  assert(m_NumBoxes > 0);
  assert(m_NumClasses > 0);
  assert(_n_grid_h > 0);
  assert(_n_grid_w > 0);
  assert(m_scale_x_y > 0);
  m_OutputSize = _n_grid_h * _n_grid_w * (m_NumBoxes * (4 + 1 + m_NumClasses));
}

int YoloV4Layer::getNbOutputs() const noexcept
{
  return 1;
}

nvinfer1::Dims YoloV4Layer::getOutputDimensions(
  int index, const nvinfer1::Dims * inputs, int nbInputDims) noexcept
{
  assert(index == 0);
  assert(nbInputDims == 1);
  return inputs[0];
}

// void YoloV4LayerV3::configure(const nvinfer1::Dims* inputDims, int nbInputs,
//                             const nvinfer1::Dims* outputDims, int nbOutputs, int maxBatchSize)
//                             noexcept
//{
//     assert(nbInputs == 1);
//     assert(inputDims != nullptr);
// }

int YoloV4Layer::initialize() noexcept
{
  return 0;
}

void YoloV4Layer::terminate() noexcept
{
}

size_t YoloV4Layer::getWorkspaceSize(int maxBatchSize) const noexcept
{
  return 0;
}

int YoloV4Layer::enqueue(
  int batchSize, const void * const * inputs, void * const * outputs, void * workspace,
  cudaStream_t stream) noexcept
{
  NV_CUDA_CHECK(cudaYoloV4Layer(
    inputs[0], outputs[0], batchSize, _n_grid_h, _n_grid_w, m_NumClasses, m_NumBoxes, m_OutputSize,
    m_new_coords, m_scale_x_y, stream));
  return 0;
}

size_t YoloV4Layer::getSerializationSize() const noexcept
{
  printf("Serialize V4\n");
  return sizeof(m_NumBoxes) + sizeof(m_NumClasses) + sizeof(_n_grid_w) + sizeof(_n_grid_h) +
         sizeof(m_OutputSize) + sizeof(m_scale_x_y) + sizeof(m_new_coords);
}

PluginFieldCollection YoloV4LayerPluginCreator::mFC{};
std::vector<PluginField> YoloV4LayerPluginCreator::mPluginAttributes;

YoloV4LayerPluginCreator::YoloV4LayerPluginCreator()
{
  mPluginAttributes.clear();

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char * YoloV4LayerPluginCreator::getPluginName() const noexcept
{
  return "YOLOV4_TRT";
}

const char * YoloV4LayerPluginCreator::getPluginVersion() const noexcept
{
  return "1.0";
}

const PluginFieldCollection * YoloV4LayerPluginCreator::getFieldNames() noexcept
{
  return &mFC;
}

IPluginV2 * YoloV4LayerPluginCreator::createPlugin(
  const char * name, const PluginFieldCollection * fc) noexcept
{
  YoloV4Layer * obj = new YoloV4Layer();
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

IPluginV2 * YoloV4LayerPluginCreator::deserializePlugin(
  const char * name, const void * serialData, size_t serialLength) noexcept
{
  // This object will be deleted when the network is destroyed, which will
  // call MishPlugin::destroy()
  YoloV4Layer * obj = new YoloV4Layer(serialData, serialLength);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

void YoloV4LayerPluginCreator::setPluginNamespace(const char * libNamespace) noexcept
{
  mNamespace = libNamespace;
}

const char * YoloV4LayerPluginCreator::getPluginNamespace() const noexcept
{
  return mNamespace.c_str();
}

}  // namespace nvinfer1
