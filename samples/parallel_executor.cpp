#include "class_timer.hpp"
#include "class_detector.h"
#include "yolo_config_parser.h"
#include <memory>
#include <thread>

#include <filesystem>
#include <iostream>
#include <sys/stat.h>
#include <iostream>
#include <boost/filesystem.hpp>
#include <omp.h>
namespace fs = boost::filesystem;

template <typename ... Args>
std::string format(const std::string& fmt, Args ... args )
{
  size_t len = std::snprintf( nullptr, 0, fmt.c_str(), args ... );
  std::vector<char> buf(len + 1);
  std::snprintf(&buf[0], len + 1, fmt.c_str(), args ... );
  return std::string(&buf[0], &buf[0] + len);
}

void
write_prediction(std::string dumpPath, std::string filename, std::vector<std::string> names, std::vector<Result> objects, int width, int height)
{
  int pos = filename.find_last_of(".");
  std::string body = filename.substr(0, pos);
  std::string dstName = body + ".txt";
  std::ofstream writing_file;
  fs::path p = dumpPath;
  fs::create_directory(p);  
  p.append(dstName);
  writing_file.open(p.string(), std::ios::out);  
  for (const auto & object : objects) {
    const auto left = object.rect.x;
    const auto top = object.rect.y;
    const auto right = std::clamp(left + object.rect.width, 0, width);
    const auto bottom = std::clamp(top + object.rect.height, 0, height);
    std::string writing_text = format("%s %f %d %d %d %d", names[object.id].c_str(), object.prob, left, top, right, bottom);
    writing_file << writing_text << std::endl;
  }
  writing_file.close();
}


int main(int argc, char** argv)
{
  gflags::SetUsageMessage(
			  "Usage : trt-yolo-app --flagfile=</path/to/config_file.txt> --<flag>=value ...");

  // parse config params
  yoloConfigParserInit(argc, argv);
  NetworkInfo yoloInfo = getYoloNetworkInfo();
  NetworkInfo yoloInfo1 = getYoloNetworkInfo1();  
  NetworkInfo yoloInfo2 = getYoloNetworkInfo2();  
  std::string directory = getDirectoryPath();
  std::string videoPath = getVideoPath();
  bool flg_save = getSaveDetections();
  std::string save_path = getSaveDetectionsPath();
  bool dont_show = get_dont_show_flg();
  const std::string dumpPath = get_dump_path();
  const bool cuda = get_cuda_flg();
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
  /*
  Config config1;
  config1.net_type = YOLOV4;
  config1.file_model_cfg = yoloInfo1.configFilePath;
  config1.file_model_weights = yoloInfo1.wtsFilePath;//"../configs/yolov7-tiny-relu-BDD100K-960x960-opt-params-mse-sparse_last.weights";		
  config1.calibration_image_list_file_txt = "../configs/calibration_images.txt";
  if (yoloInfo1.precision == "kHALF") {
    config1.inference_precison = FP16;
  }else if (yoloInfo1.precision == "kINT8") {
    config1.inference_precison = INT8;    
  } else {
    config1.inference_precison = FP32;
  }
  config1.detect_thresh = (float)get_score_thresh();
  config1.batch =  yoloInfo1.batch;
  config1.width =  yoloInfo1.width;
  config1.height=  yoloInfo1.height;
  config1.dla = yoloInfo1.dla;
  std::unique_ptr<Detector> detector1(new Detector());
  detector1->init(config1);
  */
  
  Config config2;
  config2.net_type = YOLOV4;
  config2.file_model_cfg = yoloInfo2.configFilePath;
  config2.file_model_weights = yoloInfo2.wtsFilePath;//"../configs/yolov7-tiny-relu-BDD100K-960x960-opt-params-mse-sparse_last.weights";		
  config2.calibration_image_list_file_txt = "../configs/calibration_images.txt";
  if (yoloInfo2.precision == "kHALF") {
    config2.inference_precison = FP16;
  }else if (yoloInfo2.precision == "kINT8") {
    config2.inference_precison = INT8;    
  } else {
    config2.inference_precison = FP32;
  }
  config2.detect_thresh = (float)get_score_thresh();
  config2.batch =  yoloInfo2.batch;
  config2.width =  yoloInfo2.width;
  config2.height=  yoloInfo2.height;
  config2.dla = yoloInfo2.dla;
  std::unique_ptr<Detector> detector2(new Detector());
  detector2->init(config2);
  
  std::vector<BatchResult> batch_res;
  std::vector<BatchResult> batch_res1;
  std::vector<BatchResult> batch_res2;  
  Timer timer;
  if (flg_save) {
    fs::create_directory(save_path);
    fs::path p = save_path;
    p.append("detections");
    fs::create_directory(p);
  }
  
  
  if (directory != "") {
    for (const auto & file : std::filesystem::directory_iterator(directory)) {
      std::cout << file.path() << std::endl;
      cv::Mat src = cv::imread(file.path(), cv::IMREAD_UNCHANGED);
      std::vector<cv::Mat> batch_img;
      //      batch_img.push_back(src);
      for (int b = 0; b < config.batch; b++) {
	batch_img.push_back(src);
      }      
      timer.reset();
#pragma omp parallel sections
      {
#pragma omp section
	{
	  detector->detect(batch_img, batch_res, cuda);
	}
#pragma omp section
	{
	  //detector1->detect(batch_img, batch_res, cuda);
	}	
#pragma omp section
	{
	  detector2->detect(batch_img, batch_res2, cuda);
	}
      }
      detector->segment(batch_img);
      //detector1->segment(batch_img);                
      detector2->segment(batch_img);          
      timer.out("detect");

      if (dumpPath != "not-specified") {
	fs::path p (file.path());
	std::string filename = p.filename().string();
	std::vector<std::string> names = get_names();
	write_prediction(dumpPath, filename, names, batch_res[0], src.cols, src.rows);
      }      
      //disp
      if (dont_show == true) {
	continue;
      }
      for (int i=0;i<batch_img.size();++i) {
	for (const auto &r : batch_res[i]) {
	  detector->draw_BBox(batch_img[i], r);
	}
	/*
	for (const auto &r : batch_res1[i]) {
	  detector1->draw_BBox(batch_img[i], r);
	}
	*/	
	for (const auto &r : batch_res2[i]) {
	  detector2->draw_BBox(batch_img[i], r);	  
	}	
	cv::namedWindow("image" + std::to_string(i), cv::WINDOW_NORMAL);
	cv::imshow("image"+std::to_string(i), batch_img[i]);
	int k = cv::waitKey(0);
	if (k == 32) {
	  fs::path p (file.path());
	  std::string name = p.filename().string();
	  std::cout << "Save... " << name << std::endl;
	  detector->save_image(src, "log/", name);
	  cv::waitKey(0);
	}
      }
    }
  } else if (videoPath != ""){
    std::cout << videoPath << std::endl;
    cv::VideoCapture video;
    video.open(videoPath);    
    cv::Mat frame;
    int count = 0;
    while (1) {
      video >> frame;
      if (frame.empty() == true) break;

      std::vector<cv::Mat> batch_img;
      batch_img.push_back(frame);
      timer.reset();
#pragma omp parallel sections
      {
#pragma omp section
	{
	  detector->detect(batch_img, batch_res, cuda);
	}
#pragma omp section
	{
	  //detector1->detect(batch_img, batch_res1, cuda);
	}	
#pragma omp section
	{
	  detector2->detect(batch_img, batch_res2, cuda);
	}
      }      
      detector->segment(batch_img);
      //      detector1->segment(batch_img);      
      detector2->segment(batch_img);          
      timer.out("detect");

      if (dont_show == true) {
	continue;
      }      
      //disp
      for (int i=0;i<batch_img.size();++i) {
	for (const auto &r : batch_res[i]) {
	  detector->draw_BBox(batch_img[i], r);
	}
	/*
	for (const auto &r : batch_res1[i]) {
	  detector1->draw_BBox(batch_img[i], r);
	}
	*/
	for (const auto &r : batch_res2[i]) {
	  detector2->draw_BBox(batch_img[i], r);
	}		
	//cv::namedWindow("image" + std::to_string(i), cv::WINDOW_NORMAL);
	cv::namedWindow("image" + std::to_string(i));
	cv::imshow("image"+std::to_string(i), batch_img[i]);

	if (flg_save) {
	  std::ostringstream sout;
	  sout << std::setfill('0') << std::setw(6) << count;	  
	  std::string name = "frame_" + sout.str() + ".png";
	  fs::path p = save_path;
	  p.append("detections");
	  detector->save_image(batch_img[i], p.string(), name);
	}
	if (i > 0) {
	  count++;
	}
      }
      count++;
      if (cv::waitKey(1) == 'q') break;
    }    
  }
  if (get_prof_flg()) {
    detector->dump_profiling();
  }
}
