
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:

cmake_minimum_required(VERSION 3.12)

project(gpu_playground)

#set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)

file(GLOB INC *.h)
file(GLOB SRC *.cpp)
file(GLOB CU *.cu)

find_package(CUDA QUIET REQUIRED)

set(CUDA_VERBOSE_BUILD ON)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#set(CUDA_BUILD_CUBIN ON)

include_directories(${CUDA_INCLUDE_DIRS})

# set(CUDA_NVCC_FLAGS "--compiler-options;-mtune=core2;-mfpmath=sse")
#set(CUDA_NVCC_FLAGS "--ptxas-options;--maxrregcount=128;")
set(CUDA_NVCC_FLAGS "--ptxas-options;-v;-keep")

cuda_select_nvcc_arch_flags(ARCH_FLAGS 3.5)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

# Set CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE when you want to add the same .cu
# file to multiple targets.
#set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
#set(CUDA_VERBOSE_BUILD ON)
#set(CUDA_64_BIT_DEVICE_CODE OFF)

cuda_compile(CUDA_FILES ${CU})
#cuda_compile_cubin(CUBIN_FILES ${CU})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN -D_USE_MATH_DEFINES)

if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set(CV_PATH "C:/work/Titan/thirdparty/x64/vc15/lib")
else()
    set(CV_PATH "C:/work/Titan/thirdparty/x86/vc15/lib")
endif()

find_package(OpenCV REQUIRED core highgui imgproc imgcodecs videoio PATHS ${CV_PATH})

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SRC} ${INC} ${CUDA_FILES})

set(TARGET_BIN "$<TARGET_FILE_NAME:gpu_playground>")
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

