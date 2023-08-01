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

#include "class_timer.hpp"
#include "class_detector.h"
#include "yolo_config_parser.h"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

class LightNetTensorRT : public rclcpp::Node
{
private:
  using Image = sensor_msgs::msg::Image;
public:
  LightNetTensorRT();

private:
  void on_image(Image::ConstSharedPtr msg);

  rclcpp::Subscription<Image>::SharedPtr sub_image_;
  std::unique_ptr<Detector> detector_;
  Config config_;
  bool cuda_flg_;
  std::string directory_tmp_;
};
#endif  // LIGHTNET_TRT__LIGHTNET_TRT_CORE_HPP_
