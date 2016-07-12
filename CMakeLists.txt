project(amdgpu_intercept)

add_library(amdgpu_intercept SHARED intercept.cpp)
target_include_directories(amdgpu_intercept PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/drm/include/drm ${CMAKE_CURRENT_SOURCE_DIR}/drm/amdgpu)
target_link_libraries(amdgpu_intercept PRIVATE dl drm_amdgpu)

set_property(TARGET amdgpu_intercept PROPERTY CXX_STANDARD 11)
set_property(TARGET amdgpu_intercept PROPERTY CXX_STANDARD_REQUIRED ON)
