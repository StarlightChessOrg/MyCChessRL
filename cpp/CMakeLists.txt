cmake_minimum_required(VERSION 3.18)
project(xqwl_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 若未显式指定，则用 find_package 解析到的解释器（与 conda 环境一致）
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "xqwl_core: using Python_EXECUTABLE=${Python_EXECUTABLE}")

# 可选：仍尝试从 GitHub 拉源码（国内/防火墙下可能极慢或看似卡死，默认关闭）
option(
  MYCCHESSRL_FETCH_PYBIND11
  "If pybind11 is missing, download release tarball from GitHub (needs HTTPS; can be slow)"
  OFF)

# 1) 本地源码目录（仓库已自带 cpp/pybind11-master，无需 pip / 联网）
set(_pybind11_search_paths
    "${CMAKE_CURRENT_SOURCE_DIR}/pybind11-master"
    "${CMAKE_CURRENT_SOURCE_DIR}/pybind11"
    "${CMAKE_CURRENT_SOURCE_DIR}/../pybind11"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pybind11-master")

set(_pybind11_from_subdir FALSE)
foreach(_root IN LISTS _pybind11_search_paths)
  if(EXISTS "${_root}/CMakeLists.txt")
    message(STATUS "xqwl_core: using local pybind11 at ${_root}")
    add_subdirectory("${_root}" "${CMAKE_CURRENT_BINARY_DIR}/pybind11-build")
    set(_pybind11_from_subdir TRUE)
    break()
  endif()
endforeach()

# 2) 已 pip 安装 pybind11：用当前解释器解析 CMake 包路径
if(NOT _pybind11_from_subdir)
  message(STATUS "xqwl_core: probing pybind11 via: ${Python_EXECUTABLE} -c import...")
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _p11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _p11_err
    RESULT_VARIABLE _p11_rc)
  if(_p11_rc EQUAL "0" AND EXISTS "${_p11_cmake_dir}/pybind11Config.cmake")
    message(STATUS "xqwl_core: using pip pybind11 (pybind11_DIR=${_p11_cmake_dir})")
    set(pybind11_DIR "${_p11_cmake_dir}")
    find_package(pybind11 CONFIG REQUIRED)
  elseif(MYCCHESSRL_FETCH_PYBIND11)
    message(
      STATUS
      "xqwl_core: MYCCHESSRL_FETCH_PYBIND11=ON — downloading pybind11 v2.13.6 tarball (please wait)...")
    include(FetchContent)
    FetchContent_Declare(
      pybind11 URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz)
    FetchContent_MakeAvailable(pybind11)
    message(STATUS "xqwl_core: FetchContent pybind11 done")
  else()
    message(
      FATAL_ERROR
      "xqwl_core: 未找到 pybind11。\n"
      "  最快做法（推荐）: 在与编译相同的 conda 环境中执行:\n"
      "    ${Python_EXECUTABLE} -m pip install pybind11\n"
      "  然后清空 build 目录再 cmake。\n"
      "  或: 恢复仓库内的 cpp/pybind11-master/（含顶层 CMakeLists.txt）。\n"
      "  若必须从网上拉取且网络可达 GitHub，可加:\n"
      "    -DMYCCHESSRL_FETCH_PYBIND11=ON\n"
      "  若此前 cmake 卡在「Found Python」之后，多半是自动 git clone 被墙/极慢；"
      "本仓库已默认关闭静默拉取，请先 pip install pybind11。\n"
      "  import 错误信息: ${_p11_err}")
  endif()
endif()

pybind11_add_module(xqwl_core MODULE bindings.cpp)

target_include_directories(xqwl_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

if(MSVC)
  target_compile_options(xqwl_core PRIVATE /W3 /EHsc)
else()
  target_compile_options(xqwl_core PRIVATE -Wall -Wextra -Wno-sign-compare)
endif()
