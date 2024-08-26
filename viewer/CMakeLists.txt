cmake_minimum_required(VERSION 3.1)
project(fc_viewer
  DESCRIPTION
  "Fast CD Python Bindings"
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE "Release")
set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#get fast_cd library... should really put this in deps and separate the official python bindings from this repo

add_subdirectory(./deps/fast_cd)
add_subdirectory(./deps/pybind11/)

pybind11_add_module(fc_viewer
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/viewer.cpp"
)



target_include_directories(fc_viewer PUBLIC ./deps/fast_cd/src/include/)
target_include_directories(fc_viewer PUBLIC ./src/include/)
target_include_directories(fc_viewer PUBLIC ./src/cpp/)

target_link_libraries(fc_viewer PUBLIC fast_cd)

set_target_properties(fc_viewer PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET fc_viewer PROPERTY CXX_STANDARD 17)