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

LightNetTensorRT::LightNetTensorRT(const std::string &model_cfg, const std::string &model_weights)
{
  const bool cuda = get_cuda_flg();

  // Initialize a detector
  ::Config config;

  config.net_type = YOLOV4;
  config.file_model_cfg = model_cfg;
  config.file_model_weights = model_weights;
  config.inference_precison = FP32;
  config.batch = 1;
  config.width = 1280;
  config.height = 960;
  config.dla = -1;
  detector_ = std::make_unique<::Detector>();
  detector_->init(config);
}

bool LightNetTensorRT::doInference(std::vector<cv::Mat> images)
{
  std::vector<BatchResult> batch_res;

  detector_->detect(images, batch_res, cuda);
  detector_->segment(images, "");

  for (int i = 0; i < images.size(); i++)
  {
    for (const auto &r : batch_res[i])
    {
      detector_->draw_BBox(images[i], r);
    }
  }

  return true;
}
