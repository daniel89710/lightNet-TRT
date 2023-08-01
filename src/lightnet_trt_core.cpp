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

#include "utils.hpp"

#include <rclcpp/logging.hpp>
#include <cv_bridge/cv_bridge.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <thread>

#include <filesystem>
#include <iostream>
#include <sys/stat.h>
#include <iostream>
#include <boost/filesystem.hpp>


LightNetTensorRT::LightNetTensorRT()
: rclcpp::Node("lightnet_trt")
{
  sub_image_ = this->create_subscription<sensor_msgs::msg::Image>(
    "input/image", rclcpp::QoS(1),
    std::bind(&LightNetTensorRT::on_image, this, std::placeholders::_1));

  NetworkInfo yoloInfo = getYoloNetworkInfo();
  directory_tmp_ = getDirectoryPath();
  int cam_id = getCameraID();
  // bool flg_save = getSaveDetections();
  cuda_flg_ = get_cuda_flg();

  config_.net_type = YOLOV4;
  config_.file_model_cfg = yoloInfo.configFilePath;//"../configs/yolov7-tiny-relu-BDD100K-960x960-opt-params-mse-sparse.cfg";	
  config_.file_model_weights = yoloInfo.wtsFilePath;//"../configs/yolov7-tiny-relu-BDD100K-960x960-opt-params-mse-sparse_last.weights";		
  config_.calibration_image_list_file_txt = "../configs/calibration_images.txt";
  if (yoloInfo.precision == "kHALF") {
    config_.inference_precison = FP16;
  }else if (yoloInfo.precision == "kINT8") {
    config_.inference_precison = INT8;    
  } else {
    config_.inference_precison = FP32;
  }
  config_.detect_thresh = (float)get_score_thresh();
  config_.batch = yoloInfo.batch;
  config_.width = yoloInfo.width;
  config_.height= yoloInfo.height;
  config_.dla = yoloInfo.dla;
  detector_ = std::make_unique<Detector>();
  detector_->init(config_);
}


void LightNetTensorRT::on_image(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "Received image");
  const std::string dummy_name = "";

  cv::Mat src = cv_bridge::toCvShare(msg, "bgr8")->image;

  std::vector<cv::Mat> batch_img;
  for (int b = 0; b < config_.batch; b++) {
    batch_img.push_back(src);
  }

  std::vector<BatchResult> batch_res;
  detector_->detect(batch_img, batch_res, cuda_flg_);
  detector_->segment(batch_img, dummy_name);
}
