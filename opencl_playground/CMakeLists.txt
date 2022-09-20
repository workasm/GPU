
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:

cmake_minimum_required(VERSION 3.12)

project(opencl_playground)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)

file(GLOB INC include/*.h include/*.hpp)
file(GLOB SRC *.cpp)
file(GLOB CL *.cl)

find_path(OpenCL_INCLUDE_DIRS
  NAMES
    CL/cl.h OpenCL/cl.h
  PATHS
    ENV "PROGRAMFILES(X86)"
    ENV INTELOCLSDKROOT
  PATH_SUFFIXES
    include
    OpenCL/common/inc
    "AMD APP/include"
  NO_DEFAULT_PATH)

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    find_library(OpenCL_LIBRARIES
      NAMES OpenCL
      PATHS
        ENV INTELOCLSDKROOT
      PATH_SUFFIXES
        "AMD APP/lib/x86"
        lib/x86
        lib/Win32
        OpenCL/common/lib/Win32
      NO_DEFAULT_PATH)
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    find_library(OpenCL_LIBRARIES
      NAMES OpenCL
      PATHS
        ENV INTELOCLSDKROOT
      PATH_SUFFIXES
        "AMD APP/lib/x86_64"
        lib/x86_64
        lib/x64
        OpenCL/common/lib/x64
      NO_DEFAULT_PATH)
  endif()

#find_package(OpenCL REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(. ${OpenCL_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SRC} ${INC} ${CL})

target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})

