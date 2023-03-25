cmake_minimum_required(VERSION 3.1) 
project(lightNet-TRT VERSION 1.0)            
enable_language(CUDA)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wno-write-strings ${OpenMP_CXX_FLAGS} -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -g -Wno-write-strings")
#set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")



include_directories(/usr/local/cuda/include)
find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)

file(GLOB_RECURSE sources modules/*.hpp modules/*.cpp modules/*.h modules/*.cu extra/*.h module/preprocess.cu)
add_library(detector SHARED ${sources})
target_include_directories(detector PRIVATE extra/ modules/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(detector nvinfer nvinfer_plugin nvparsers gflags "stdc++fs")


add_executable(lightNet-TRT samples/sample_detector.cpp ${sources})


target_include_directories(lightNet-TRT PRIVATE modules/ extra/)
target_link_libraries(lightNet-TRT  ${OpenCV_LIBS} nvinfer nvinfer_plugin nvparsers gflags boost_system boost_filesystem "stdc++fs" "stdc++")
