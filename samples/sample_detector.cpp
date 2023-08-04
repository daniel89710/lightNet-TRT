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


void
write_label(std::string outputPath, std::string filename, std::vector<std::string> names, std::vector<Result> objects, int width, int height)
{
  int pos = filename.find_last_of(".");
  std::string body = filename.substr(0, pos);
  std::string dstName = body + ".txt";
  std::ofstream writing_file;
  fs::path p = outputPath;
  fs::create_directory(p);  
  p.append(dstName);
  writing_file.open(p.string(), std::ios::out);
  std::cout << "Write" << p.string() << std::endl;
  for (const auto & object : objects) {
    const auto left = object.rect.x;
    const auto top = object.rect.y;
    const auto right = std::clamp(left + object.rect.width, 0, width);
    const auto bottom = std::clamp(top + object.rect.height, 0, height);
    double x = (left + right) / 2.0 / (double)width;
    double y = (top + bottom) / 2.0 / (double)height;
    double w = (right - left) / (double)width;
    double h = (bottom - top) / (double)height;    
    std::string writing_text = format("%d %f %f %f %f", object.id, x, y, w, h);
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
  
  
  if (directory != "") {
    for (const auto & file : std::filesystem::directory_iterator(directory)) {
      std::cout << file.path() << std::endl;
      cv::Mat src = cv::imread(file.path(), cv::IMREAD_UNCHANGED);
      fs::path p (file.path());
      std::string name = p.filename().string();
      
      std::vector<cv::Mat> batch_img;
      //      batch_img.push_back(src);
      for (int b = 0; b < config.batch; b++) {
	batch_img.push_back(src);
      }      
      detector->detect(batch_img, batch_res, cuda);
      detector->segment(batch_img, name);    

      if (dumpPath != "not-specified") {
	fs::path p (file.path());
	std::string filename = p.filename().string();
	std::vector<std::string> names = get_names();
	write_prediction(dumpPath, filename, names, batch_res[0], src.cols, src.rows);
      }
      if (outputPath != "not-specified") {
	fs::path p (file.path());
	std::string filename = p.filename().string();
	std::vector<std::string> names = get_names();
	write_label(outputPath, filename, names, batch_res[0], src.cols, src.rows);
      }      
      if (target != "") {
	std::vector<std::string> names = get_names();
	for (int i=0;i<batch_img.size();++i) {
	  for (const auto &r : batch_res[i]) {
	    int id = r.id;
	    if (names[id] == target) {
	      fs::path p = save_path;
	      p.append("target");
	      std::cout << "Find " << target << " -> save into " << p.string() << std::endl;
	      detector->save_image(batch_img[i], p.string(), name);	    
	    }
	  }
	}
      }
      //disp
      if (dont_show == true) {
	continue;
      }
      for (int i=0;i<batch_img.size();++i) {
	for (const auto &r : batch_res[i]) {
	  //	  std::cout <<"batch "<<i<< " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
	  detector->draw_BBox(batch_img[i], r);
	}
	// cv::namedWindow("image" + std::to_string(i), cv::WINDOW_NORMAL);
	// cv::imshow("image"+std::to_string(i), batch_img[i]);
	int k = cv::waitKey(0);
	if (k == 32) {
	  std::cout << "Save... " << name << std::endl;
	  detector->save_image(src, "log/", name);
	  cv::waitKey(0);
	}
      }
    }
  } else if (videoPath != "" || cam_id != -1){
    std::cout << videoPath << std::endl;
    cv::VideoCapture video;
    if (cam_id != -1) {
      video.open(cam_id);
    } else {
      video.open(videoPath);
    }
    cv::Mat frame;
    int count = 0;
    while (1) {
      video >> frame;
      if (frame.empty() == true) break;

      std::vector<cv::Mat> batch_img;
      batch_img.push_back(frame);
      detector->detect(batch_img, batch_res, cuda);
      detector->segment(batch_img, "");    

      //disp
      for (int i=0;i<batch_img.size();++i) {
	for (const auto &r : batch_res[i]) {
	  //	  std::cout <<"batch "<<i<< " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
	  detector->draw_BBox(batch_img[i], r);
	}
	if (!dont_show) {
	  cv::namedWindow("image" + std::to_string(i), cv::WINDOW_NORMAL);
	  cv::imshow("image"+std::to_string(i), batch_img[i]);
	}
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
