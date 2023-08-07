#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>

void blobFromImageGpu(
  float * dst, unsigned char * src, int d_w, int d_h, int d_c, int s_w, int s_h, int s_c,
  float norm, cudaStream_t stream);
