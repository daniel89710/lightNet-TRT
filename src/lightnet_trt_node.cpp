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

#include "lightnet_trt_node.hpp"

LightNetTensorRTNode::LightNetTensorRTNode(const rclcpp::NodeOptions &node_options)
  : Node("tensorrt_lightnet", node_options)
{
  using std::placeholders::_1;
  using std::chrono_literals::operator""ms;

  // Initialize a detector
  ::Config config = loadConfig();

  RCLCPP_INFO(this->get_logger(), "Start loading YOLO");
  RCLCPP_INFO(this->get_logger(), "Model Config: %s", config.file_model_cfg.c_str());
  RCLCPP_INFO(this->get_logger(), "Model Weights: %s", config.file_model_weights.c_str());
  RCLCPP_INFO(this->get_logger(), "Input size: (%d, %d)", config.width, config.height);
  lightnet_trt_ = std::make_unique<LightNetTensorRT>(config);
  RCLCPP_INFO(this->get_logger(), "Finished loading YOLO");

  image_sub_ = image_transport::create_subscription(
    this, "~/in/image", std::bind(&LightNetTensorRTNode::onImage, this, _1), "raw",
    rmw_qos_profile_sensor_data);

  image_pub_ = image_transport::create_publisher(this, "~/out/image");
}

::Config LightNetTensorRTNode::loadConfig()
{
  const std::string model_cfg = declare_parameter<std::string>("model_cfg");
  const std::string model_weights = declare_parameter<std::string>("model_weights");
  const int width = declare_parameter<int>("width");
  const int height = declare_parameter<int>("height");

  // Initialize a detector
  ::Config config;

  config.net_type = YOLOV4;
  config.file_model_cfg = model_cfg;
  config.file_model_weights = model_weights;
  config.inference_precison = FP32;
  config.batch = 1;
  config.width = width;
  config.height = height;
  config.dla = -1;

  return config;
}

void LightNetTensorRTNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "Image received");
  cv_bridge::CvImagePtr in_image_ptr;
  in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

  if (!lightnet_trt_->doInference({in_image_ptr->image}))
  {
    RCLCPP_WARN(this->get_logger(), "Inference failed.");
    return;
  }
  image_pub_.publish(in_image_ptr->toImageMsg());
  RCLCPP_INFO(this->get_logger(), "Inference succeeded");
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(LightNetTensorRTNode)
