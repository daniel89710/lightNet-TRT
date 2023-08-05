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

#ifndef LIGHTNET_TRT__UTILS_HPP_
#define LIGHTNET_TRT__UTILS_HPP_

#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sensor_msgs/msg/camera_info.hpp"
#include "geometry_msgs/msg/transform.hpp"

#include <utility>

Eigen::Matrix3f get_intrinsic_matrix(const sensor_msgs::msg::CameraInfo & camera_info);
Eigen::Matrix3f resize_intrinsic_matrix(const Eigen::Matrix3f & intrinsic, const int resized_width, const int resized_height);

class IPM
{
private:
  using Pair = std::pair<float, float>;
public:
  IPM(const Eigen::Matrix3f & intrinsic, const Eigen::Matrix4f & extrinsic,
	  const Pair & roi_x, const Pair & roi_y);
  void run(const cv::Mat & img, cv::Mat & output_img);
private:
  cv::Mat perspective_transform_;
  int img_width_;
  int img_height_;
};

#endif  // LIGHTNET_TRT__UTILS_HPP_
