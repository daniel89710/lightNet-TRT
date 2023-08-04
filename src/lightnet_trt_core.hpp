// Copyright 2023 TIER IV
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

#ifndef LIGHTNET_TRT__LIGHTNET_TRT_CORE_HPP_
#define LIGHTNET_TRT__LIGHTNET_TRT_CORE_HPP_

#include "class_detector.h"
#include "yolo_config_parser.h"

#include <opencv2/opencv.hpp>

#include <vector>

class LightNetTensorRT
{
public:
  LightNetTensorRT(const ::Config &config);

  void doInference(const std::vector<cv::Mat> & images, std::vector<cv::Mat> & masks);

  std::unique_ptr<Detector> detector_;
  const bool cuda = false;
};
#endif  // LIGHTNET_TRT__LIGHTNET_TRT_CORE_HPP_
