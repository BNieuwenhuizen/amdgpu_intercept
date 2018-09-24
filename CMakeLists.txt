cmake_minimum_required(VERSION 3.12)
project(amdgpu_intercept)

add_library(amdgpu_intercept SHARED intercept.cpp)
target_include_directories(amdgpu_intercept PRIVATE ${LIBDRM_INSTALL_DIR}/include/libdrm)
target_link_libraries(amdgpu_intercept PRIVATE dl drm_amdgpu)

set_property(TARGET amdgpu_intercept PROPERTY CXX_STANDARD 11)
set_property(TARGET amdgpu_intercept PROPERTY CXX_STANDARD_REQUIRED ON)
