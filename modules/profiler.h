#ifndef YOLO_PROFILER_H
#define YOLO_PROFILER_H

#include <NvInfer.h>
#include <vector>
#include <string>
#include <iostream>
#include <map>

struct LayerInfo
{
  int in_c;
  int out_c;
  int w;
  int h;
  int k;      
  int stride;
  int groups;
  nvinfer1::LayerType type;
};


class SimpleProfiler : public nvinfer1::IProfiler
{
public:
    struct Record
    {
      float time{0};
      int count{0};
      float min_time{-1.0};
      int index;
    };
    /*
     *  Construct a profiler
     *  name:           Name of the profiler
     *  src_profilers:  Optionally initialize profiler with data of one or more other profilers
     *                  This is usefull for aggregating results of different profilers
     *                  Aggregation will sum all runtime periods and all invokations for each reported
     *                  layer of all given profilers
     */
    SimpleProfiler(std::string name,
        const std::vector<SimpleProfiler>& src_profilers = std::vector<SimpleProfiler>());

    void reportLayerTime(const char* layerName, float ms) noexcept override;

    void setProfDict(nvinfer1::ILayer *layer) noexcept;
    LayerInfo *getILayerFromName(std::string name) noexcept;
    friend std::ostream& operator<<(std::ostream& out, SimpleProfiler& value);

private:
  std::string m_name;
  std::map<std::string, Record> m_profile;
  int m_index;
  std::map<std::string, LayerInfo>m_layer_dict;
};



#endif /* JETNET_PROFILER_H */
