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

LightNetTensorRTNode::LightNetTensorRTNode(const rclcpp::NodeOptions & node_options)
: Node("tensorrt_lightnet", node_options), ipm_projector_(nullptr), base_frame_("base_link")
{
  transform_listener_ = std::make_shared<tier4_autoware_utils::TransformListener>(this);

  // Pamareters
  roi_x_.first = this->declare_parameter<double>("roi_x_min");
  roi_x_.second = this->declare_parameter<double>("roi_x_max");
  roi_y_.first = this->declare_parameter<double>("roi_y_min");
  roi_y_.second = this->declare_parameter<double>("roi_y_max");

  // Initialize a detector
  config_ = loadConfig();

  RCLCPP_INFO(this->get_logger(), "Start loading YOLO");
  RCLCPP_INFO(this->get_logger(), "Model Config: %s", config_.file_model_cfg.c_str());
  RCLCPP_INFO(this->get_logger(), "Model Weights: %s", config_.file_model_weights.c_str());
  RCLCPP_INFO(this->get_logger(), "Input size: (%d, %d)", config_.width, config_.height);
  lightnet_trt_ = std::make_unique<LightNetTensorRT>(config_);
  RCLCPP_INFO(this->get_logger(), "Finished loading YOLO");

  image_sub_ = image_transport::create_subscription(
    this, "~/in/image", [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) { onImage(msg); },
    "raw", rmw_qos_profile_sensor_data);
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "~/in/camera_info", rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) { onCameraInfo(msg); });

  image_pub_ = image_transport::create_publisher(this, "~/out/image");
  image_projected_pub_ = image_transport::create_publisher(this, "~/out/image_projected");
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

void LightNetTensorRTNode::onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
  if (ipm_projector_ != nullptr) return;
  const Eigen::Matrix3f intrinsic = get_intrinsic_matrix(*msg);
  const Eigen::Matrix3f intrinsic_resized =
    resize_intrinsic_matrix(intrinsic, config_.width, config_.height);

  const std::string camera_frame = msg->header.frame_id;
  geometry_msgs::msg::TransformStamped::ConstSharedPtr tf_base2cam_ptr =
    transform_listener_->getLatestTransform(base_frame_, camera_frame);
  if (!tf_base2cam_ptr) return;

  const Eigen::Matrix4f extrinsic =
    tf2::transformToEigen(tf_base2cam_ptr->transform).matrix().cast<float>();
  ipm_projector_ = std::make_unique<IPM>(intrinsic_resized, extrinsic, roi_x_, roi_y_);
  return;
}

void LightNetTensorRTNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "Image received");
  cv_bridge::CvImagePtr in_image_ptr;
  in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

  std::vector<cv::Mat> masks;
  lightnet_trt_->doInference({in_image_ptr->image}, masks);
  RCLCPP_INFO(this->get_logger(), "Inference succeeded");

  // Publish result
  cv_bridge::CvImage out_image;
  out_image.header = msg->header;
  out_image.encoding = sensor_msgs::image_encodings::BGR8;
  out_image.image = masks[0];
  image_pub_.publish(out_image.toImageMsg());

  // Post processing
  if (!ipm_projector_) {
    RCLCPP_WARN(this->get_logger(), "IPM projector is not initialized.");
    return;
  }
  RCLCPP_WARN(this->get_logger(), "IPM projector running.");

  cv::Mat projected_mask;
  ipm_projector_->run(masks[0], projected_mask);

  // Publish result
  cv_bridge::CvImage out_projected_image;
  out_projected_image.header = msg->header;
  out_projected_image.encoding = sensor_msgs::image_encodings::BGR8;
  out_projected_image.image = projected_mask;
  image_projected_pub_.publish(out_projected_image.toImageMsg());
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(LightNetTensorRTNode)
