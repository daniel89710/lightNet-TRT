/**
   MIT License

   Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
   *
   */

#include "yolo_config_parser.h"

#include <assert.h>
#include <iostream>

DEFINE_bool(dont_show, false,
            "[OPTIONAL] Flag to off screen");

DEFINE_string(d, "",
              "Directory Path, "
              "Directory Path");

DEFINE_string(v, "",
              "Video Path, "
              "Video Path");

DEFINE_int64(cam_id, -1, "CAMERA_ID");

DEFINE_string(network_type, "not-specified",
              "[REQUIRED] Type of network architecture. Choose from yolov2, yolov2-tiny, "
              "yolov3 and yolov3-tiny");
DEFINE_string(config_file_path, "not-specified", "[REQUIRED] Darknet cfg file");
DEFINE_string(wts_file_path, "not-specified", "[REQUIRED] Darknet weights file");
DEFINE_string(config_file_path1, "not-specified", "[REQUIRED] Darknet cfg file");
DEFINE_string(wts_file_path1, "not-specified", "[REQUIRED] Darknet weights file");
DEFINE_string(config_file_path2, "not-specified", "[REQUIRED] Darknet cfg file");
DEFINE_string(wts_file_path2, "not-specified", "[REQUIRED] Darknet weights file");
DEFINE_string(labels_file_path, "../configs/bdd100k.names", "[REQUIRED] Object class labels file");
DEFINE_string(precision, "kFLOAT",
              "[OPTIONAL] Inference precision. Choose from kFLOAT, kHALF and kINT8.");
DEFINE_string(deviceType, "kGPU",
              "[OPTIONAL] The device that this layer/network will execute on. Choose from kGPU and kDLA(only for kHALF).");
DEFINE_string(calibration_table_path, "not-specified",
              "[OPTIONAL] Path to pre-generated calibration table. If flag is not set, a new calib "
              "table <network-type>-<precision>-calibration.table will be generated");
DEFINE_string(engine_file_path, "not-specified",
              "[OPTIONAL] Path to pre-generated engine(PLAN) file. If flag is not set, a new "
              "engine <network-type>-<precision>-<batch-size>.engine will be generated");
DEFINE_string(input_blob_name, "data",
              "[OPTIONAL] Name of the input layer in the tensorRT engine file");

DEFINE_string(
	      test_images, "data/test_images.txt",
	      "[REQUIRED] Text file containing absolute paths or filenames of all the images to be "
	      "used for inference. If only filenames are provided, their corresponding source directory "
	      "has to be provided through 'test_images_path' flag");
DEFINE_string(test_images_path, "not-specified",
              "[OPTIONAL] absolute source directory path of the list of images supplied through "
              "'test_images' flag");
DEFINE_string(calibration_images, "data/calibration_images.txt",
	      "[OPTIONAL] Text file containing absolute paths or filenames of calibration images. "
              "Flag required if precision is kINT8 and there is not pre-generated calibration "
              "table. If only filenames are provided, their corresponding source directory has to "
              "be provided through 'calibration_images_path' flag");
DEFINE_string(calibration_images_path, "not-specified",
              "[OPTIONAL] absolute source directory path of the list of images supplied through "
              "'calibration_images' flag");
DEFINE_uint64(batch_size, 1, "[OPTIONAL] Batch size for the inference engine.");
DEFINE_uint64(width, 0, "[OPTIONAL] width for the inference engine.");
DEFINE_uint64(height, 0, "[OPTIONAL] height for the inference engine.");
DEFINE_double(thresh, 0.2, "[OPTIONAL] thresh");

DEFINE_bool(save_detections, false,
            "[OPTIONAL] Flag to save images overlayed with objects detected.");
DEFINE_string(save_detections_path, "outputs/",
              "[OPTIONAL] Path where the images overlayed with bounding boxes are to be saved");

DEFINE_bool(prof, false,
            "[OPTIONAL] Flag to profile layer by layer");

DEFINE_string(dump, "not-specified",
              "[OPTIONAL] Path to dump predictions for mAP calculation");

DEFINE_string(output, "not-specified",
              "[OPTIONAL] Path to output predictions for pseudo-labeling");

DEFINE_bool(mp, false,
            "[OPTIONAL] Flag to multi-precision");
DEFINE_int64(dla, -1, "[OPTIONAL] DLA");
DEFINE_int64(dla1, -1, "[OPTIONAL] DLA");
DEFINE_int64(dla2, -1, "[OPTIONAL] DLA");
DEFINE_bool(cuda, false,
            "[OPTIONAL] Flag to cuda preprcessing");

DEFINE_string(target, "",
              "[OPTIONAL] Target Finder");

static bool isFlagDefault(std::string flag) { return flag == "not-specified" ? true : false; }

static bool networkTypeValidator(const char* flagName, std::string value)
{
  /*if (((FLAGS_network_type) == "yolov2") || ((FLAGS_network_type) == "yolov2-tiny")
    || ((FLAGS_network_type) == "yolov3") || ((FLAGS_network_type) == "yolov3-tiny"))
    return true;

    else
    std::cout << "Invalid value for --" << flagName << ": " << value << std::endl;
  */
  return false;
}

static bool precisionTypeValidator(const char* flagName, std::string value)
{
  /*  if ((FLAGS_precision == "kFLOAT") || (FLAGS_precision == "kINT8")
      || (FLAGS_precision == "kHALF"))
      return true;
      else
      std::cout << "Invalid value for --" << flagName << ": " << value << std::endl;*/
  return false;
}

static bool verifyRequiredFlags()
{
  /* assert(!isFlagDefault(FLAGS_network_type)
     && "Type of network is required and is not specified.");
     assert(!isFlagDefault(FLAGS_config_file_path)
     && "Darknet cfg file path is required and not specified.");
     assert(!isFlagDefault(FLAGS_wts_file_path)
     && "Darknet weights file is required and not specified.");
     assert(!isFlagDefault(FLAGS_labels_file_path) && "Lables file is required and not specified.");
     assert((FLAGS_wts_file_path.find(".weights") != std::string::npos)
     && "wts file not recognised. File needs to be of '.weights' format");
     assert((FLAGS_config_file_path.find(".cfg") != std::string::npos)
     && "config file not recognised. File needs to be of '.cfg' format");
     if (!(networkTypeValidator("network_type", FLAGS_network_type)
     && precisionTypeValidator("precision", FLAGS_precision)))
     return false;
  */
  return true;
}

void yoloConfigParserInit(int argc, char** argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  assert(verifyRequiredFlags());

  FLAGS_calibration_images_path
    = isFlagDefault(FLAGS_calibration_images_path) ? "" : FLAGS_calibration_images_path;
  FLAGS_test_images_path = isFlagDefault(FLAGS_test_images_path) ? "" : FLAGS_test_images_path;
  
  if (isFlagDefault(FLAGS_engine_file_path))
    {
      int npos = FLAGS_wts_file_path.find(".weights");
      assert(npos != std::string::npos
	     && "wts file file not recognised. File needs to be of '.weights' format");
      std::string dataPath = FLAGS_wts_file_path.substr(0, npos);
      FLAGS_engine_file_path = dataPath + "-" + FLAGS_precision + "-" + FLAGS_deviceType + "-batch"
	+ std::to_string(FLAGS_batch_size) + ".engine";
    }

  if (isFlagDefault(FLAGS_calibration_table_path))
    {
      int npos = FLAGS_wts_file_path.find(".weights");

      assert(npos != std::string::npos
	     && "wts file file not recognised. File needs to be of '.weights' format");
      std::string dataPath = FLAGS_wts_file_path.substr(0, npos);
      FLAGS_calibration_table_path = dataPath + "-calibration.table";
    }
}

std::string getDirectoryPath(void)
{
  return FLAGS_d;
}

std::string getVideoPath(void)
{
  return FLAGS_v;
}

int getCameraID(void)
{
  return FLAGS_cam_id;
}

NetworkInfo getYoloNetworkInfo()
{
  return NetworkInfo{FLAGS_network_type,     FLAGS_config_file_path, FLAGS_wts_file_path,
		       FLAGS_labels_file_path, FLAGS_precision,        FLAGS_deviceType,
		       FLAGS_calibration_table_path, FLAGS_engine_file_path, FLAGS_input_blob_name,
		       "", FLAGS_batch_size, FLAGS_width, FLAGS_height, FLAGS_dla};
}

NetworkInfo getYoloNetworkInfo1()
{
  return NetworkInfo{FLAGS_network_type,     FLAGS_config_file_path1, FLAGS_wts_file_path1,
		       FLAGS_labels_file_path, FLAGS_precision,        FLAGS_deviceType,
		       FLAGS_calibration_table_path, FLAGS_engine_file_path, FLAGS_input_blob_name,
		       "", FLAGS_batch_size, FLAGS_width, FLAGS_height, FLAGS_dla1};
}

NetworkInfo getYoloNetworkInfo2()
{
  return NetworkInfo{FLAGS_network_type,     FLAGS_config_file_path2, FLAGS_wts_file_path2,
		       FLAGS_labels_file_path, FLAGS_precision,        FLAGS_deviceType,
		       FLAGS_calibration_table_path, FLAGS_engine_file_path, FLAGS_input_blob_name,
		       "", FLAGS_batch_size, FLAGS_width, FLAGS_height, FLAGS_dla2};
}

std::vector<std::string>
get_names(void)
{
  std::string filename = FLAGS_labels_file_path;
  std::vector<std::string> names;
  if (filename != "not-specified") {
    names = loadListFromTextFile(filename);    
  }
  return names;
}
bool getSaveDetections()
{
  if (FLAGS_save_detections)
    assert(!isFlagDefault(FLAGS_save_detections_path)
	   && "save_detections path has to be set if save_detections is set to true");
  return FLAGS_save_detections;
}

std::string getSaveDetectionsPath() { return FLAGS_save_detections_path; }

bool
get_dont_show_flg(void)
{
  return FLAGS_dont_show;
}


bool
get_prof_flg(void)
{
  return FLAGS_prof;
}


std::string
get_dump_path(void)
{
  return FLAGS_dump;
}

double
get_score_thresh(void)
{
  return FLAGS_thresh;
}


bool
get_multi_precision_flg(void)
{
  return FLAGS_mp;
}


bool
get_cuda_flg(void)
{
  return FLAGS_cuda;
}

std::string
get_output_path(void)
{
  return FLAGS_output;
}


std::string
get_target_label(void)
{
  return FLAGS_target;
}
