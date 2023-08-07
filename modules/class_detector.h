#ifndef CLASS_DETECTOR_H_
#define CLASS_DETECTOR_H_

#include "API.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>

struct Result
{
  int id = -1;
  float prob = 0.f;
  cv::Rect rect;
};

using BatchResult = std::vector<Result>;

enum ModelType { YOLOV3, YOLOV4, YOLOV5 };

enum Precision { INT8 = 0, FP16, FP32 };

struct Config
{
  std::string file_model_cfg = "configs/yolov4.cfg";

  std::string file_model_weights = "configs/yolov4.weights";

  float detect_thresh = 0.9;

  ModelType net_type = YOLOV4;

  Precision inference_precison = FP32;

  int gpu_id = 0;

  std::string calibration_image_list_file_txt = "configs/calibration_images.txt";
  int batch = 0;
  int width = 0;
  int height = 0;
  int dla = -1;
};

class API Detector
{
public:
  explicit Detector();

  ~Detector();

  void init(const Config & config);
  void save_image(cv::Mat & img, const std::string & dir, const std::string & name);
  void segment(const std::vector<cv::Mat> & mat_image, std::string filename);
  void get_mask(std::vector<cv::Mat> & mask_results);
  void detect(
    const std::vector<cv::Mat> & mat_image, std::vector<BatchResult> & vec_batch_result,
    const bool cuda);
  void dump_profiling(void);
  void draw_BBox(cv::Mat & img, Result result);

private:
  Detector(const Detector &);
  const Detector & operator=(const Detector &);
  class Impl;
  Impl * _impl;
};

#endif  // !CLASS_QH_DETECTOR_H_
