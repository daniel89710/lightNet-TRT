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

#include <algorithm>
#include <iostream>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <thread>
#include <vector>
#include <filesystem>
#include <sys/stat.h>
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

#endif  // LIGHTNET_TRT__UTILS_HPP_
