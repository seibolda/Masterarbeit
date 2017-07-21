cmake_minimum_required(VERSION 2.8)
project(Pottslab_cuda C CXX)

#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 2.4.9 REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    --use_fast_math --compiler-options -Wall -std=c++11 -Wno-deprecated-gpu-targets -Xcompiler -fPIC
)

set(SOURCE_FILES
        src/main.cu
        src/util/helper.cu
        src/util/CudaBuffer.cu
        src/Image.cpp
        src/PottsSolver.cpp
)

cuda_add_executable(Pottslab_cuda ${SOURCE_FILES})

target_include_directories(Pottslab_cuda
    PUBLIC
        ${OpenCV_INCLUDE_DIR}
        include
)

target_link_libraries(Pottslab_cuda
        ${OpenCV_LIBS}
        ${CUDA_CUBLAS_LIBRARIES}
        opencv_highgui
        opencv_core
)