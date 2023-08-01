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

#include "class_timer.hpp"
#include "class_detector.h"
#include "yolo_config_parser.h"

#include "utils.hpp"

#include <rclcpp/logging.hpp>

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
  std::string directory = getDirectoryPath();
  std::string videoPath = getVideoPath();
  int cam_id = getCameraID();
  bool flg_save = getSaveDetections();
  std::string save_path = getSaveDetectionsPath();
  bool dont_show = get_dont_show_flg();
  const std::string dumpPath = get_dump_path();
  const bool cuda = get_cuda_flg();
  const std::string target = get_target_label();
  const std::string outputPath = get_output_path();
  Config config;
  config.net_type = YOLOV4;
  config.file_model_cfg = yoloInfo.configFilePath;//"../configs/yolov7-tiny-relu-BDD100K-960x960-opt-params-mse-sparse.cfg";	
  config.file_model_weights = yoloInfo.wtsFilePath;//"../configs/yolov7-tiny-relu-BDD100K-960x960-opt-params-mse-sparse_last.weights";		
  config.calibration_image_list_file_txt = "../configs/calibration_images.txt";
  if (yoloInfo.precision == "kHALF") {
    config.inference_precison = FP16;
  }else if (yoloInfo.precision == "kINT8") {
    config.inference_precison = INT8;    
  } else {
    config.inference_precison = FP32;
  }
  config.detect_thresh = (float)get_score_thresh();
  config.batch =  yoloInfo.batch;
  config.width =  yoloInfo.width;
  config.height=  yoloInfo.height;
  config.dla = yoloInfo.dla;
  std::unique_ptr<Detector> detector(new Detector());
  detector->init(config);
  std::vector<BatchResult> batch_res;
  if (flg_save) {
    fs::create_directory(save_path);
    fs::path p = save_path;
    p.append("detections");
    fs::create_directory(p);
  }
  
}


void LightNetTensorRT::on_image(const sensor_msgs::msg::Image::ConstSharedPtr)
{
  RCLCPP_INFO(this->get_logger(), "Received image");
  //cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
  //cv::imshow("view", image);
  //cv::waitKey(30);
}