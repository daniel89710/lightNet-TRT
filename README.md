# LightNet-TRT: High-Efficiency and Real-Time CNN Implementation on Edge AI

LightNet-TRT is a CNN implementation optimized for edge AI devices that combines the advantages of LightNet and TensorRT. LightNet is a lightweight and high-performance neural network framework designed for edge devices, while TensorRT is a high-performance deep learning inference engine developed by NVIDIA for optimizing and running deep learning models on GPUs. LightNet-TRT uses the Network Definition API provided by TensorRT to integrate LightNet into TensorRT, allowing it to run efficiently and in real-time on edge devices.





## Key Improvements

### 2:4 Structured Sparsity

LightNet-TRT utilizes 2:4 structured sparsity to further optimize the network. 2:4 structured sparsity means that every 2x2 block of weights in a convolutional layer is reduced to a single 4-bit value, resulting in a 75% reduction in the number of weights. This technique allows the network to use fewer weights and computations while maintaining accuracy.

![Sparsity](https://developer-blogs.nvidia.com/ja-jp/wp-content/uploads/sites/6/2022/06/2-4-structured-sparse-matrix.png "sparsity")

### NVDLA Execution

LightNet-TRT also supports the execution of the neural network on the NVIDIA Deep Learning Accelerator (NVDLA), a free and open architecture that provides high performance and low power consumption for deep learning inference on edge devices. By using NVDLA, LightNet-TRT can further improve the efficiency and performance of the network on edge devices.

![NVDLA](https://i0.wp.com/techgrabyte.com/wp-content/uploads/2019/09/Nvidia-Open-Source-Its-Deep-Learning-Inference-Compiler-NVDLA-2.png?w=768&ssl=1 "NVDLA")


### Multi-Precision Quantization

In addition to partial quantizatiBy writing it in CFG, you can set the precision for each layer of your CNNon, LightNet-TRT also supports multi-precision quantization, which allows the network to use different precision for weights and activations. By using mixed precision, LightNet-TRT can further reduce the memory usage and computational requirements of the network while still maintaining accuracy. By writing it in CFG, you can set the precision for each layer of your CNN

![Quantization](https://developer-blogs.nvidia.com/wp-content/uploads/2021/07/qat-training-precision.png "Quantization")



### Multitask Execution (Detection/Segmentation)

LightNet-TRT also supports multitask execution, allowing the network to perform both object detection and segmentation tasks simultaneously. This enables the network to perform multiple tasks efficiently on edge devices, saving computational resources and power.

[![](https://img.youtube.com/vi/TmlW-b_t3sQ/0.jpg)](https://www.youtube.com/watch?v=TmlW-b_t3sQ)

## Installation

### Requirements

-   CUDA11.0 or higher
-   TensorRT 8.0 or higher
-   OpenCV 3.0 or higher

### Steps

1.  Clone the repository.
    
```shell
$ git clone https://github.com/daniel89710/lightNet-TRT.git
$ cd lightNet-TRT
```
	
2.  Install libraries.
						    
```shell
$ sudo apt update
$ sudo apt install libgflags-dev
$ sudo apt install libboost-all-dev
```
										    
3.  Compile the TensorRT implementation.
											    
```shell
$ mkdir build
$ cmake ../
$ make -j
```
															    
																
## Usage

### Converting a lightNet model to a TensorRT engine
						
Build FP32 engine
```shell
$ ./lightNet-TRT --flagfile ../configs/lightNet-BDD100K-det-semaseg-1280x960.txt --precision kFLOAT
```

Build FP16(HALF) engine
```shell
$ ./lightNet-TRT --flagfile ../configs/lightNet-BDD100K-det-semaseg-1280x960.txt --precision kHALF
```

Build INT8 engine 
(You need to prepare a list for calibration in "configs/calibration_images.txt".)
```shell
$ ./lightNet-TRT --flagfile ../configs/lightNet-BDD100K-det-semaseg-1280x960.txt --precision kINT8
```

Build DLA engine 
```shell
$ ./lightNet-TRT --flagfile ../configs/lightNet-BDD100K-det-semaseg-1280x960.txt --precision [kHALF/kINT8] --dla [0/1]
```

### Inference with the TensorRT engine

Inference from images
```shell
$ ./lightNet-TRT --flagfile ../configs/lightNet-BDD100K-det-semaseg-1280x960.txt --precision [kFLOAT/kHALF/kINT8] {--dla [0/1]} --d DIRECTORY
```

Inference from images
```shell
$ ./lightNet-TRT --flagfile ../configs/lightNet-BDD100K-det-semaseg-1280x960.txt --precision [kFLOAT/kHALF/kINT8] {--dla [0/1]} --d VIDEO
```

## Implementation

LightNet-TRT is built on the LightNet framework and integrates with TensorRT using the Network Definition API. The implementation is based on the following repositories:

-   LightNet: [https://github.com/daniel89710/lightNet](https://github.com/daniel89710/lightNet)
-   TensorRT: [https://github.com/NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)
-   NVIDIA DeepStream SDK: [https://github.com/NVIDIA-AI-IOT/deepstream\_reference\_apps/tree/restructure](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/restructure)
-   YOLO-TensorRT: [https://github.com/enazoe/yolo-tensorrt](https://github.com/enazoe/yolo-tensorrt)

## Conclusion

LightNet-TRT is a powerful and efficient implementation of CNNs using Edge AI. With its advanced features and integration with TensorRT, it is an excellent choice for real-time object detection and semantic segmentation applications on edge devices.


