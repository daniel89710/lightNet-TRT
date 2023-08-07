#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include "preprocess.h"

#define BLOCK 512

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d;
    d.x = x;
    d.y = y;
    d.z = 1;
    return d;
}


__global__ void SimpleblobFromImageKernel(int N, float* dst_img, unsigned char* src_img, 
				       int dst_h, int dst_w, int src_h, int src_w, 
				    float stride_h, float stride_w, float norm)
{
  int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

  if (index >= N) return;
  int chan = 3; 
  int w = index % dst_w;
  int h = index / dst_w;
  float centroid_h, centroid_w;
  int c;
  centroid_h = stride_h * (float)(h + 0.5); 
  centroid_w = stride_w * (float)(w + 0.5);
  int src_h_idx = lroundf(centroid_h)-1;
  int src_w_idx = lroundf(centroid_w)-1;
  if (src_h_idx<0){src_h_idx=0;}
  if (src_w_idx<0){src_w_idx=0;}  
  index = chan * src_w_idx + chan* src_w * src_h_idx;

  for (c = 0; c < chan; c++) {
    int dst_index = w + (dst_w*h) + (dst_w*dst_h*c);              
    dst_img[dst_index] = (float)src_img[index+c]*norm;
  }
}

void blobFromImageGpu(float *dst, unsigned char*src, int d_w, int d_h, int d_c,
			 int s_w, int s_h, int s_c, float norm, cudaStream_t stream)
{
  int N =  d_w * d_h;
  float stride_h = (float)s_h / (float)d_h;
  float stride_w = (float)s_w / (float)d_w;

  SimpleblobFromImageKernel<<<cuda_gridsize(N), BLOCK, 0, stream>>>(N, dst, src, 
							      d_h, d_w,
							      s_h, s_w,
							      stride_h, stride_w, norm);

}

