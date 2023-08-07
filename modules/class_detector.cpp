#include "class_detector.h"

#include "class_yolo_detector.hpp"

#include <string>
#include <vector>

class Detector::Impl
{
public:
  Impl() {}

  ~Impl() {}

  YoloDectector _detector;
};

Detector::Detector()
{
  _impl = new Impl();
}

Detector::~Detector()
{
  if (_impl) {
    delete _impl;
    _impl = nullptr;
  }
}

void Detector::init(const Config & config)
{
  _impl->_detector.init(config);
}

void Detector::dump_profiling(void)
{
  _impl->_detector.dump_profiling();
}

void Detector::detect(
  const std::vector<cv::Mat> & mat_image, std::vector<BatchResult> & vec_batch_result,
  const bool cuda)
{
  _impl->_detector.detect(mat_image, vec_batch_result, cuda);
}

void Detector::save_image(cv::Mat & img, const std::string & dir, const std::string & name)
{
  _impl->_detector.save_image(img, dir, name);
}

void Detector::segment(const std::vector<cv::Mat> & mat_image, std::string filename)
{
  _impl->_detector.segment(mat_image, filename);
}

void Detector::get_mask(std::vector<cv::Mat> & mask_results)
{
  _impl->_detector.get_mask(mask_results);
}

void Detector::draw_BBox(cv::Mat & img, Result result)
{
  _impl->_detector.draw_BBox(img, result);
}
