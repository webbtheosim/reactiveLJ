cmake_minimum_required(VERSION 3.24)
project(ReactiveAHAddon LANGUAGES CXX)
find_package(hoomd CONFIG REQUIRED)
pybind11_add_module(_reactive_ah MODULE NO_EXTRAS
    module.cc ReactiveAHForceCompute.cc
    ReactiveAHForceComputeGPU.cc ReactiveAHForceComputeGPU.cu)
target_link_libraries(_reactive_ah PRIVATE HOOMD::_md)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80-real;80-virtual")
endif()
set_target_properties(_reactive_ah PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../code")
