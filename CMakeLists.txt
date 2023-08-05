# cmake_minimum_required(VERSION 3.1)
cmake_minimum_required(VERSION 3.14)
project(lightnet_trt VERSION 1.0)         

enable_language(CUDA) # not supported on CMAKE 3.14

find_package(OpenMP REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr") # Added for CMAKE 3.14
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wno-unused-function -Wno-maybe-uninitialized -Wno-narrowing -Wno-misleading-indentation -Wno-unused-but-set-variable -Wno-unused-variable -Wno-return-type -Wno-deprecated-declarations -Wno-write-strings -Wno-reorder -Wno-sign-compare -Wno-unused-parameter -Wno-unused-value -Wno-uninitialized ${OpenMP_CXX_FLAGS} -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -g -Wno-write-strings")
#set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

include_directories(/usr/local/cuda/include)
find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)

file(GLOB_RECURSE sources modules/*.hpp modules/*.cpp modules/*.h modules/*.cu extra/*.h module/preprocess.cu)
add_library(detector SHARED ${sources})

target_include_directories(detector PRIVATE extra/ modules/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(detector ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvparsers gflags "stdc++fs")

add_executable(lightnet_trt samples/sample_detector.cpp ${sources})

target_include_directories(lightnet_trt PRIVATE modules/ extra/)
target_link_libraries(lightnet_trt ${CUDA_LIBRARIES} ${OpenCV_LIBS} nvinfer nvinfer_plugin nvparsers gflags boost_system boost_filesystem "stdc++fs" "stdc++")


# ROS 2 package
if(BUILD_ROS2_LIGHTNET_TRT)
  find_package(autoware_cmake REQUIRED)
  autoware_package()

  find_package(Eigen3 REQUIRED)

  # build nodelet
  # TODO

  # build node
  ament_auto_add_executable(lightnet_trt_node
    src/main.cpp
    src/lightnet_trt_node.cpp
    src/lightnet_trt_core.cpp
    src/utils.cpp
  )
  ament_target_dependencies(lightnet_trt_node)
  target_include_directories(lightnet_trt_node PRIVATE modules/ extra/)
  target_link_libraries(lightnet_trt_node detector)
  target_link_libraries(lightnet_trt_node ${CUDA_LIBRARIES} ${OpenCV_LIBS} Eigen3::Eigen nvinfer nvinfer_plugin nvparsers gflags boost_system boost_filesystem "stdc++fs" "stdc++")

  ament_auto_package(INSTALL_TO_SHARE
    launch
    configs
    ros_config
  )
endif()
