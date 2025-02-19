cmake_minimum_required(VERSION 3.28)
project(MySharedLibrary C)

file(GLOB C_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/data/cc/*.c)
add_library(model SHARED ${C_SOURCES})

set_target_properties(model PROPERTIES LIBRARY_OUTPUT_DIRECTORY data/)
set_target_properties(model PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/data/cc/*.c)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(model PRIVATE -O0 -fno-inline -fPIC -g1)
else()
    target_compile_options(model PRIVATE -O3 -ffast-math -fPIC -ffunction-sections -fdata-sections)
    target_link_options(model PRIVATE -fuse-ld=lld -Wl,--gc-sections)
endif()