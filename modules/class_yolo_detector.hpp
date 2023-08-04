#ifndef CLASS_YOLO_DETECTOR_HPP_
#define CLASS_YOLO_DETECTOR_HPP_

#include <opencv2/opencv.hpp>
#include "ds_image.h"
#include "trt_utils.h"
#include "yolo.h"
#include "yolov2.h"
#include "yolov3.h"
#include "yolov4.h"
#include "yolov5.h"
#include "yolo_config_parser.h"

#include <experimental/filesystem>
#include <fstream>
#include <string>
#include <chrono>
#include <stdio.h>  /* defines FILENAME_MAX */
#include <iostream>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#include "class_detector.h"
#include "class_timer.hpp"

static int GLOBAL_COUNTER = 0;

class YoloDectector
{
public:
  YoloDectector()
  {

  }
  ~YoloDectector()
  {

  }

  void init(const Config &config)
  {
    _config = config;

    this->set_gpu_id(_config.gpu_id);

    this->parse_config();

    this->build_net();
  }

  void save_image(cv::Mat &img, const std::string &dir, const std::string &name)
  {
    fs::path p = dir;
    p.append(name);
    std::string dst = p.string();
    std::cout << "##Save " << dst << std::endl;
    cv::imwrite(dst, img);
  }
  
  void draw_BBox(cv::Mat &img, Result result) {
    std::stringstream stream;
    uint32_t *colormap = _p_net->get_detection_colormap();
    int id = result.id;
    if (colormap) {
      cv::rectangle(img, result.rect, cv::Scalar(colormap[3*id+2], colormap[3*id+1], colormap[3*id+0]), 4);
    } else {
      cv::rectangle(img, result.rect, cv::Scalar(255, 0, 0), 2);
    }
    std::vector<std::string> names = _p_net->get_detection_names(0);
    if (names.size()) {
      stream << std::fixed << std::setprecision(2) << names[id] << "  " << result.prob;
    } else {
      stream << std::fixed << std::setprecision(2) << "id:" << id << "  score:" << result.prob;
    }

    if (colormap) {
      cv::putText(img, stream.str(), cv::Point(result.rect.x, result.rect.y - 5), 0, 0.5, cv::Scalar(colormap[3*id+2], colormap[3*id+1], colormap[3*id+0]), 1);
    } else {
      cv::putText(img, stream.str(), cv::Point(result.rect.x, result.rect.y - 5), 0, 0.5, cv::Scalar(255, 0, 0), 1);
    } 
  }

  void segment(const std::vector<cv::Mat> &vec_image, std::string filename)
  {
    bool flg_save = getSaveDetections();
    std::string save_path = getSaveDetectionsPath();

    for (uint32_t i = 0; i < vec_image.size(); ++i) {
      auto curImage = vec_image.at(i);
      auto segmentation = _p_net->apply_argmax(i);
      auto mask = _p_net->get_colorlbl(segmentation);
      int height = curImage.rows;
      int width = curImage.cols;
      
      for (uint32_t j = 0; j < mask.size(); j++) {
				cv::Mat resized;
				
				cv::resize(mask[j], resized, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);	
				cv::addWeighted(vec_image[i], 1.0, resized, 0.5, 0.0, vec_image[i]);
				
				cv::namedWindow("mask" + std::to_string(j), cv::WINDOW_NORMAL);
				cv::imshow("mask"+std::to_string(j), mask[j]);
				if (flg_save) {
					fs::path p = save_path;
					p.append("segmentation");
					fs::create_directory(p);
					p.append(std::to_string(j));
					fs::create_directory(p);
					if (0) {
						//get filenames;
					} else {
						std::ostringstream sout;
						sout << std::setfill('0') << std::setw(6) << GLOBAL_COUNTER;
						//cv::resize(mask[j], resized, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
						if (filename == "") {
							std::string filename = "frame_" + sout.str() + ".png";
						}
						save_image(resized, p.string(), filename);
					}
				}
				auto depthmap = _p_net->get_depthmap(segmentation);
				for (uint32_t j = 0; j < depthmap.size(); j++) {
					cv::imshow("depthmap"+std::to_string(j), depthmap[j]);
				}
			}
			if (i > 0) {
				GLOBAL_COUNTER++;
			}
		}
    GLOBAL_COUNTER++;
  } 

  void get_mask(std::vector<cv::Mat> &mask_results)
  {
		auto segmentation = _p_net->apply_argmax(0);
		mask_results = _p_net->get_colorlbl(segmentation);
    GLOBAL_COUNTER++;
  }

  void dump_profiling(){
    _p_net->print_profiling();    
  }
  
  void detect(const std::vector<cv::Mat>	&vec_image,
	      std::vector<BatchResult> &vec_batch_result, const bool cuda)
  {
    cv::Mat trtInput;
    std::vector<DsImage> vec_ds_images;
    vec_batch_result.clear();
    vec_batch_result.resize(vec_image.size());
    for (const auto &img:vec_image)
      {
	vec_ds_images.emplace_back(img, _vec_net_type[_config.net_type], _p_net->getInputH(), _p_net->getInputW());
      }
    

    if (cuda) {
      //std::cout << "Preprocess on GPU" << std::endl;
      _p_net->preprocess_gpu(vec_image[0].data, vec_image[0].cols, vec_image[0].rows);
      trtInput.data = NULL;
    } else {
      //      std::cout << "Preprocess on CPU" << std::endl;
      trtInput = blobFromDsImages(vec_ds_images, _p_net->getInputH(),_p_net->getInputW());
    }
    if (get_prof_flg()) {
      _p_net->doProfiling(trtInput.data, vec_ds_images.size());
    } else {
      _p_net->doInference(trtInput.data, vec_ds_images.size());
    }

    for (uint32_t i = 0; i < vec_ds_images.size(); ++i)
      {
	auto curImage = vec_ds_images.at(i);
	auto binfo = _p_net->decodeDetections(i, curImage.getImageHeight(), curImage.getImageWidth());
	//	auto segmentation = _p_net->apply_argmax(i, curImage.getImageHeight(), curImage.getImageWidth());
	auto remaining = nmsAllClasses(_p_net->getNMSThresh(),
				       binfo,
				       _p_net->getNumClasses(),
				       _vec_net_type[_config.net_type]);
	if (remaining.empty())
	  {
	    continue;
	  }
	std::vector<Result> vec_result(0);
	for (const auto &b : remaining)
	  {
	    Result res;
	    res.id = b.label;
	    res.prob = b.prob;
	    const int x = b.box.x1;
	    const int y = b.box.y1;
	    const int w = b.box.x2 - b.box.x1;
	    const int h = b.box.y2 - b.box.y1;
	    res.rect = cv::Rect(x, y, w, h);
	    vec_result.push_back(res);
	  }
	vec_batch_result[i] = vec_result;
      }

  }

private:

	void set_gpu_id(const int id = 0)
	{
		cudaError_t status = cudaSetDevice(id);
		if (status != cudaSuccess)
		{
			std::cout << "gpu id :" + std::to_string(id) + " not exist !" << std::endl;
			assert(0);
		}
	}

	void parse_config()
	{
		_yolo_info.networkType = _vec_net_type[_config.net_type];
		_yolo_info.configFilePath = _config.file_model_cfg;
		_yolo_info.wtsFilePath = _config.file_model_weights;
		_yolo_info.precision = _vec_precision[_config.inference_precison];
		_yolo_info.deviceType = "kGPU";
		auto npos = _yolo_info.wtsFilePath.find(".weights");
		assert(npos != std::string::npos
			&& "wts file file not recognised. File needs to be of '.weights' format");
		_yolo_info.data_path = _yolo_info.wtsFilePath.substr(0, npos);
		_yolo_info.calibrationTablePath = _yolo_info.data_path + "-calibration.table";
		_yolo_info.inputBlobName = "data";

		_yolo_info.batch = _config.batch;
		_yolo_info.width = _config.width;
		_yolo_info.height = _config.height;
		_yolo_info.dla = _config.dla;  		

		_infer_param.printPerfInfo = false;
		_infer_param.printPredictionInfo = false;
		_infer_param.calibImages = _config.calibration_image_list_file_txt;
		_infer_param.calibImagesPath = "";
		_infer_param.probThresh = _config.detect_thresh;
		_infer_param.nmsThresh = 0.5;
		_infer_param.prof = get_prof_flg();		
	}

	void build_net()
	{
		if (_config.net_type == YOLOV3) 
		{
			_p_net = std::unique_ptr<Yolo>{ new YoloV3(_yolo_info, _infer_param) };
		}
		else if( _config.net_type == YOLOV4)
		{
			_p_net = std::unique_ptr<Yolo>{ new YoloV4(_yolo_info,_infer_param) };
		}
		else if (_config.net_type == YOLOV5)
		{
			_p_net = std::unique_ptr<Yolo>{ new YoloV5(_yolo_info,_infer_param) };
		}
		else
		{
			assert(false && "Unrecognised network_type.");
		}
	}

private:
	Config _config;
	NetworkInfo _yolo_info;
	InferParams _infer_param;
	std::vector<std::string> _vec_net_type{ "yolov3","yolov4","yolov5" };
	std::vector<std::string> _vec_precision{ "kINT8","kHALF","kFLOAT" };
	std::unique_ptr<Yolo> _p_net = nullptr;
	Timer _m_timer;
};


#endif
