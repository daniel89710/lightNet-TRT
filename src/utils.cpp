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

#include "utils.hpp"

Eigen::Matrix3f get_intrinsic_matrix(const sensor_msgs::msg::CameraInfo & camera_info)
{
	Eigen::Matrix3f intrinsic_matrix;
	intrinsic_matrix << camera_info.k[0], camera_info.k[1], camera_info.k[2],
		camera_info.k[3], camera_info.k[4], camera_info.k[5],
		camera_info.k[6], camera_info.k[7], camera_info.k[8];
	return intrinsic_matrix;
}

Eigen::Matrix3f resize_intrinsic_matrix(const Eigen::Matrix3f & intrinsic, const int resized_width, const int resized_height)
{
	Eigen::Matrix3f resized_intrinsic;
	const float width_ratio = resized_width / 2.0 / intrinsic(0, 2);
	const float height_ratio = resized_height / 2.0 / intrinsic(1, 2);
	resized_intrinsic << intrinsic(0, 0) * width_ratio, 0, resized_width / 2,
		0, intrinsic(1, 1) * height_ratio, resized_height / 2,
		0, 0, 1;
	return resized_intrinsic;
}

// TODO: Implement this in a smarter way
IPM::IPM(const Eigen::Matrix3f & intrinsic, const Eigen::Matrix4f & extrinsic)
{
	// Define points on the ground plane in the world coordinate system
	// 50 x 20 m rectangle
	const Eigen::Vector4f dst_p1{55, -10, 0, 1};
	const Eigen::Vector4f dst_p2{55, 10, 0, 1};
	const Eigen::Vector4f dst_p3{5, 10, 0, 1};
	const Eigen::Vector4f dst_p4{5, -10, 0, 1};

	// Convert the points from the world coordinate system to the camera coordinate system
	const Eigen::Matrix4f extrinsic_inv = extrinsic.inverse();
	Eigen::Vector3f src_p1 = (extrinsic_inv * dst_p1).head<3>();
	Eigen::Vector3f src_p2 = (extrinsic_inv * dst_p2).head<3>();
	Eigen::Vector3f src_p3 = (extrinsic_inv * dst_p3).head<3>();
	Eigen::Vector3f src_p4 = (extrinsic_inv * dst_p4).head<3>();

	// Project the points from 3D camera coordinate system to 2D image plane
	src_p1 = intrinsic * src_p1; src_p1 /= src_p1(2);
	src_p2 = intrinsic * src_p2; src_p2 /= src_p2(2);
	src_p3 = intrinsic * src_p3; src_p3 /= src_p3(2);
	src_p4 = intrinsic * src_p4; src_p4 /= src_p4(2);

	// Create source and destination points for cv::getPerspectiveTransform
	const std::vector<cv::Point2f> src_pts = {
		cv::Point2f(src_p1(0), src_p1(1)),
		cv::Point2f(src_p2(0), src_p2(1)),
		cv::Point2f(src_p3(0), src_p3(1)),
		cv::Point2f(src_p4(0), src_p4(1))
	};
	const std::vector<cv::Point2f> dst_pts = {
		cv::Point2f(dst_p3(1) * 12 + 320, dst_p3(0) * 12),
		cv::Point2f(dst_p4(1) * 12 + 320, dst_p4(0) * 12),
		cv::Point2f(dst_p1(1) * 12 + 320, dst_p1(0) * 12),
		cv::Point2f(dst_p2(1) * 12 + 320, dst_p2(0) * 12)
	};

	perspective_transform_ = cv::getPerspectiveTransform(src_pts, dst_pts);
}

void IPM::run(const cv::Mat & img, cv::Mat & output_img)
{
	cv::warpPerspective(img, output_img, perspective_transform_, cv::Size(640, 640));
}