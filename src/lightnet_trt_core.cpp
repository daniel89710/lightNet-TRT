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

#include "lightnet_trt_core.hpp"

LightNetTensorRT::LightNetTensorRT(const ::Config &config)
{
  const bool cuda = get_cuda_flg();

  detector_ = std::make_unique<::Detector>();
  detector_->init(config);
}

void LightNetTensorRT::doInference(const std::vector<cv::Mat> & images, std::vector<cv::Mat> & masks)
{
  std::vector<BatchResult> batch_res;
  detector_->detect(images, batch_res, cuda);
  detector_->get_mask(masks);
}
