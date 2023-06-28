
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:

cmake_minimum_required(VERSION 3.12)

project(gpu_playground)

#include(clang_cuda_compile.cmake)
include(x_cuda_compile.cmake)

#set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)

#enable_language(CUDA)

file(GLOB INC *.h)
file(GLOB SRC *.cpp)
file(GLOB CU *.cu)

set(CUDA_64_BIT_DEVICE_CODE ON)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(CUDA 11.4 REQUIRED)

set(CUDA_VERBOSE_BUILD ON)
#set(CUDA_BUILD_CUBIN OFF)

include_directories(${CUDA_INCLUDE_DIRS})

# set(CUDA_NVCC_FLAGS "--compiler-options;-mtune=core2;-mfpmath=sse")
#set(CUDA_NVCC_FLAGS "--ptxas-options;--maxrregcount=128;")
set(CUDA_NVCC_FLAGS "-std=c++17;--ptxas-options;-v;-keep;-allow-unsupported-compiler;-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")

# clang this will pass '-v' option to ptxas assembler
#set(CUDA_NVCC_FLAGS "-Xcuda-ptxas;-v;-Xcuda-ptxas;--opt-level=3;-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG -D_DLL -Xclang --dependent-lib=msvcrt -D_MD")

cuda_select_nvcc_arch_flags(ARCH_FLAGS 3.5)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS} -Wno-deprecated-gpu-targets)

x_cuda_compile(CUDA_FILES ${CU})
#cuda_compile_cubin(CUBIN_FILES ${CU})
#cuda_compile(CUDA_FILES ${CU})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN -D_USE_MATH_DEFINES -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)

message(STATUS "--- ${CU}")

if(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
    set(CV_PATH "C:/work/Titan/thirdparty/x64/vc15/lib")
else()
    set(CV_PATH "C:/work/Titan/thirdparty/x86/vc15/lib")
endif()

find_package(OpenCV REQUIRED core highgui imgproc imgcodecs videoio PATHS ${CV_PATH})

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SRC} ${INC} ${CUDA_FILES})

set(TARGET_BIN "$<TARGET_FILE_NAME:gpu_playground>")
# NOTEL: this only worls with 'cubin' not exe files
#add_custom_command(TARGET ${PROJECT_NAME}
#    POST_BUILD
#    COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvdisasm -ndf ${TARGET_BIN}
#    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
#)

add_custom_command(TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/cuobjdump.exe -sass ${TARGET_BIN}
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)

#add_custom_command(TARGET ${PROJECT_NAME}
#    POST_BUILD
#    COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/cuda-memcheck.exe --print-level info ${TARGET_BIN}
#    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
#    COMMENT "=== running cuda-memcheck ==="
#)

target_link_libraries(${PROJECT_NAME} ${CUDA_CUDART_LIBRARY} ${OpenCV_LIBS})


# Add a special target to clean nvcc generated files.
# CUDA_BUILD_CLEAN_TARGET()

