cmake_minimum_required(VERSION 3.18)
project(xqwl_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

set(_pybind11_root "${CMAKE_CURRENT_SOURCE_DIR}/../../pybind11-master")
if(EXISTS "${_pybind11_root}/CMakeLists.txt")
  add_subdirectory("${_pybind11_root}" pybind11-build)
else()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _p11_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  find_package(pybind11 CONFIG REQUIRED HINTS "${_p11_dir}")
endif()

pybind11_add_module(xqwl_core MODULE bindings.cpp)

target_include_directories(xqwl_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

if(MSVC)
  target_compile_options(xqwl_core PRIVATE /W3 /EHsc)
else()
  target_compile_options(xqwl_core PRIVATE -Wall -Wextra -Wno-sign-compare)
endif()
