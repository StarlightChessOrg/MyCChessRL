cmake_minimum_required(VERSION 3.18)
project(xqwl_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# 1) 本地源码目录（任选其一放入即可离线编译）
set(_pybind11_search_paths
    "${CMAKE_CURRENT_SOURCE_DIR}/pybind11"
    "${CMAKE_CURRENT_SOURCE_DIR}/../pybind11"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pybind11-master")

set(_pybind11_from_subdir FALSE)
foreach(_root IN LISTS _pybind11_search_paths)
  if(EXISTS "${_root}/CMakeLists.txt")
    add_subdirectory("${_root}" "${CMAKE_CURRENT_BINARY_DIR}/pybind11-build")
    set(_pybind11_from_subdir TRUE)
    break()
  endif()
endforeach()

# 2) 已 pip 安装 pybind11：用当前解释器解析 CMake 包路径
if(NOT _pybind11_from_subdir)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _p11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _p11_err
    RESULT_VARIABLE _p11_rc)
  if(_p11_rc EQUAL "0" AND EXISTS "${_p11_cmake_dir}/pybind11Config.cmake")
    set(pybind11_DIR "${_p11_cmake_dir}")
    find_package(pybind11 CONFIG REQUIRED)
  else()
    # 3) 无本地目录且未 pip：自动拉取官方 pybind11（需能访问 GitHub）
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(pybind11)
  endif()
endif()

pybind11_add_module(xqwl_core MODULE bindings.cpp)

target_include_directories(xqwl_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

if(MSVC)
  target_compile_options(xqwl_core PRIVATE /W3 /EHsc)
else()
  target_compile_options(xqwl_core PRIVATE -Wall -Wextra -Wno-sign-compare)
endif()
