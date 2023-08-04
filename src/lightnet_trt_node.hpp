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

#ifndef LIGHTNET_TRT__LIGHTNET_TRT_NODE_HPP_
#define LIGHTNET_TRT__LIGHTNET_TRT_NODE_HPP_

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include "lightnet_trt_core.hpp"

#include <sensor_msgs/msg/image.hpp>

#include <cv_bridge/cv_bridge.h>

class LightNetTensorRTNode : public rclcpp::Node
{
public:
  explicit LightNetTensorRTNode(const rclcpp::NodeOptions &node_options);

private:
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  ::Config loadConfig();

  image_transport::Publisher image_pub_;
  image_transport::Subscriber image_sub_;
  std::unique_ptr<LightNetTensorRT> lightnet_trt_;
};
#endif  // LIGHTNET_TRT__LIGHTNET_TRT_NODE_HPP_
