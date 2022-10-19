cmake_minimum_required(VERSION 3.10)
project(amdgpu_intercept)

add_library(amdgpu_intercept SHARED intercept.cpp)
target_include_directories(amdgpu_intercept PRIVATE /usr/include/libdrm)
target_link_libraries(amdgpu_intercept PRIVATE dl)

set_property(TARGET amdgpu_intercept PROPERTY CXX_STANDARD 11)
set_property(TARGET amdgpu_intercept PROPERTY CXX_STANDARD_REQUIRED ON)
