//#include "profiler.h"
#include "profiler.h"
#include <iomanip>

using namespace nvinfer1;


SimpleProfiler::SimpleProfiler(std::string name,
                               const std::vector<SimpleProfiler>& src_profilers) :
    m_name(name)
{
  float total_time = 0.0;
  m_index = 0;
  for (const auto& src_profiler : src_profilers) {
    for (const auto& rec : src_profiler.m_profile) {
      auto it = m_profile.find(rec.first);

      if (it == m_profile.end()) {
	m_profile.insert(rec);
      } else {
	it->second.time += rec.second.time;
	it->second.count += rec.second.count;
	total_time += rec.second.time;
      }
    }
  }
}

void SimpleProfiler::reportLayerTime(const char* layerName, float ms) noexcept
{

  m_profile[layerName].count++;
  m_profile[layerName].time += ms;
  if (m_profile[layerName].min_time == -1.0) {
    m_profile[layerName].min_time = ms;
    m_profile[layerName].index = m_index;
    m_index++;    
  } else if (m_profile[layerName].min_time > ms) {
    m_profile[layerName].min_time = ms;
  }
}

void SimpleProfiler::setProfDict(nvinfer1::ILayer *layer) noexcept
{
  std::string name = layer->getName();
  auto t = layer->getType();
  (void)t; // Added by Koji Minoda
  m_layer_dict[name];
  m_layer_dict[name].type = layer->getType();
  if (layer->getType() == nvinfer1::LayerType::kCONVOLUTION) {
    nvinfer1::IConvolutionLayer* conv =  (nvinfer1::IConvolutionLayer*)layer;
    nvinfer1::ITensor* in = layer->getInput(0);
    nvinfer1::Dims dim_in = in->getDimensions();
    nvinfer1::ITensor* out = layer->getOutput(0);
    nvinfer1::Dims dim_out = out->getDimensions();
    nvinfer1::Dims k_dims = conv->getKernelSizeNd();
    nvinfer1::Dims s_dims = conv->getStrideNd();
    int groups = conv->getNbGroups();
    int stride = s_dims.d[0];
    int kernel = k_dims.d[0];
    m_layer_dict[name].in_c = dim_in.d[0];
    m_layer_dict[name].out_c = dim_out.d[0];
    m_layer_dict[name].w= dim_in.d[2];
    m_layer_dict[name].h = dim_in.d[1];
    m_layer_dict[name].k = kernel;;
    m_layer_dict[name].stride = stride;
    m_layer_dict[name].groups = groups;    
    
  }
}

LayerInfo *SimpleProfiler::getILayerFromName(std::string name) noexcept
{
  LayerInfo *linfo = NULL;
  /*
  if (m_layer_dict.find(name) != m_layer_dict.end()) {
    linfo = &m_layer_dict[name];
  }
  */
  for (auto it = m_layer_dict.begin(); it != m_layer_dict.end(); it++) {
    std::string key = it->first;
    if (!name.find(key)) {
      linfo = &m_layer_dict[key];
    }
  }
  return linfo;
}

std::ostream& operator<<(std::ostream& out, SimpleProfiler& value)
{
  out << "========== " << value.m_name << " profile ==========" << std::endl;
  float totalTime = 0;
  std::string layerNameStr = "Operation";

  int maxLayerNameLength = static_cast<int>(layerNameStr.size());
  for (const auto& elem : value.m_profile)
    {
      totalTime += elem.second.time;
      maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
    }

  auto old_settings = out.flags();
  auto old_precision = out.precision();
  // Output header
  {
    out << std::setw(maxLayerNameLength) << layerNameStr << " ";
    out << std::setw(12) << "Runtime, "
	<< "%"
	<< " ";
    out << std::setw(12) << "Invocations"
	<< " ";
    out << std::setw(12) << "Runtime[ms]" ;
    out << std::setw(12) << "Min Runtime[ms]" << std::endl;	
  }
  int index = value.m_index;
  for (int i = 0; i < index; i++) {
    for (const auto& elem : value.m_profile){
	if (elem.second.index == i) {
	  out << i << ":";
	  out << std::setw(maxLayerNameLength) << elem.first << ",";
	  out << std::setw(12) << std::fixed << std::setprecision(1) << (elem.second.time * 100.0F / totalTime) << "%"
	      << ",";
	  out << std::setw(12) << elem.second.count << ",";
	  out << std::setw(12) << std::fixed << std::setprecision(2) << elem.second.time << ", ";
	  out << std::setw(12) << std::fixed << std::setprecision(2) << elem.second.min_time << std::endl;
	}
      }
  }
  out.flags(old_settings);
  out.precision(old_precision);
  out << "========== " << value.m_name << " total runtime = " << totalTime << " ms ==========" << std::endl;
  out << "Conv Profile" << std::endl;
  out << "index, w, h, in_c, out_c, k, stride, groups, mintime[us], name" << std::endl;

  for (int i = 0; i < index; i++) {
    for (const auto& elem : value.m_profile){
	if (elem.second.index == i) {
	  LayerInfo* linfo  = value.getILayerFromName(elem.first);
	  if (linfo) {
	    if (linfo->type == nvinfer1::LayerType::kCONVOLUTION) {
	      out <<  std::setw(4) << std::fixed << i << ", ";
	      out <<  std::setw(4) << std::fixed << linfo->w << ", ";
	      out <<  std::setw(4) << std::fixed << linfo->h << ", ";
	      out <<  std::setw(4) << std::fixed << linfo->in_c << ", ";
	      out <<  std::setw(4) << std::fixed << linfo->out_c << ", ";
	      out <<  std::setw(4) << std::fixed << linfo->k << ", ";
	      out <<  std::setw(4) << std::fixed << linfo->stride << ", ";
	      out <<  std::setw(4) << std::fixed << linfo->groups << ", "; 	      	      	      	      
	    
	  
	      
	      out  <<  std::setw(10) << std::setprecision(4) << elem.second.min_time *1000 << ", ";
	      out  <<  std::setw(6) << elem.first << "," << std::endl;
	    }
	  }
	}
    }
  }

  std::vector<int> chan{8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384};
  //k=1 conv
  int kernel = 1;
  out << "###Conv Kernel(1)" <<std::endl;
  out << "     ," << std::setw(4);  
  for (int out_c : chan) {    
    out <<  std::setw(10) << ","  << out_c;
  }
  out << std::endl;
  for (int in_c : chan) {
    out << "in_c, " << std::setw(4) << std::fixed << in_c << ", ";
    for (int out_c : chan) {
      bool flg  = false;
      for (int i = 0; i < index && flg == false; i++) {
	for (const auto& elem : value.m_profile){
	  if (elem.second.index == i) {
	    LayerInfo* linfo  = value.getILayerFromName(elem.first);
	    if (linfo && linfo->k == kernel && in_c == linfo->in_c && out_c == linfo->out_c) {
	      if (linfo->type == nvinfer1::LayerType::kCONVOLUTION) {
		out  <<  std::setw(10) << std::setprecision(4) << elem.second.min_time *1000;
		flg = true;
		break;
	      }
	    }
	  }
	}
      }
      out << ", ";      
    }
    out   << std::endl;    
  }
  kernel = 3;
  out << "###Conv Kernel(" << kernel << ")" <<std::endl;
  out << "     ," << std::setw(4);  
  for (int out_c : chan) {    
    out <<  std::setw(10) << ","  << out_c;
  }
  out << std::endl;
  for (int in_c : chan) {
    out << "in_c, " << std::setw(4) << std::fixed << in_c << ", ";
    for (int out_c : chan) {
      bool flg  = false;
      for (int i = 0; i < index && flg == false; i++) {
	for (const auto& elem : value.m_profile){
	  if (elem.second.index == i) {
	    LayerInfo* linfo  = value.getILayerFromName(elem.first);
	    if (linfo && linfo->k == kernel && in_c == linfo->in_c && out_c == linfo->out_c) {
	      if (linfo->type == nvinfer1::LayerType::kCONVOLUTION) {
		out  <<  std::setw(10) << std::setprecision(4) << elem.second.min_time *1000;
		flg = true;
		break;
	      }
	    }
	  }
	}
      }
      out << ", ";      
    }
    out   << std::endl;    
  }

  kernel = 5;
  out << "###Conv Kernel(" << kernel << ")" <<std::endl;
  out << "     ," << std::setw(4);  
  for (int out_c : chan) {    
    out <<  std::setw(10) << ","  << out_c;
  }
  out << std::endl;
  for (int in_c : chan) {
    out << "in_c, " << std::setw(4) << std::fixed << in_c << ", ";
    for (int out_c : chan) {
      bool flg  = false;
      for (int i = 0; i < index && flg == false; i++) {
	for (const auto& elem : value.m_profile){
	  if (elem.second.index == i) {
	    LayerInfo* linfo  = value.getILayerFromName(elem.first);
	    if (linfo && linfo->k == kernel && in_c == linfo->in_c && out_c == linfo->out_c) {
	      if (linfo->type == nvinfer1::LayerType::kCONVOLUTION) {
		out  <<  std::setw(10) << std::setprecision(4) << elem.second.min_time *1000;
		flg = true;
		break;
	      }
	    }
	  }
	}
      }
      out << ", ";      
    }
    out   << std::endl;    
  }
  kernel = 7;
  out << "###Conv Kernel(" << kernel << ")" <<std::endl;
  out << "     ," << std::setw(4);  
  for (int out_c : chan) {    
    out <<  std::setw(10) << ","  << out_c;
  }
  out << std::endl;
  for (int in_c : chan) {
    out << "in_c, " << std::setw(4) << std::fixed << in_c << ", ";
    for (int out_c : chan) {
      bool flg  = false;
      for (int i = 0; i < index && flg == false; i++) {
	for (const auto& elem : value.m_profile){
	  if (elem.second.index == i) {
	    LayerInfo* linfo  = value.getILayerFromName(elem.first);
	    if (linfo && linfo->k == kernel && in_c == linfo->in_c && out_c == linfo->out_c) {
	      if (linfo->type == nvinfer1::LayerType::kCONVOLUTION) {
		out  <<  std::setw(10) << std::setprecision(4) << elem.second.min_time *1000;
		flg = true;
		break;
	      }
	    }
	  }
	}
      }
      out << ", ";      
    }
    out   << std::endl;    
  }         
  return out;
}


