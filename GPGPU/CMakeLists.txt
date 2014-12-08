cmake_minimum_required(VERSION 2.6)

# project name
set(PROJECT_NAME GPGPU)
project(${PROJECT_NAME})

# bin,src,common and include dir
set(BIN ${PROJECT_SOURCE_DIR}/bin)
set(SRC ${PROJECT_SOURCE_DIR}/src)
set(COM ${PROJECT_SOURCE_DIR}/../Common/src)
set(INC ${PROJECT_SOURCE_DIR}/../Common/include)

# executable under ./bin
set(EXECUTABLE_OUTPUT_PATH ${BIN})

# find files
file(GLOB_RECURSE source_files ${SRC}/*)


# Finding libs
find_package(CUDA REQUIRED)

# set some options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_12,code=sm_12 -ccbin g++")


cuda_add_executable(
    ${PROJECT_NAME}
    ${source_files}
    ${COM}
    ${INC}
)

# include
include_directories(
    ${SRC}
    ${common_files}
)

# libs
target_link_libraries(
    ${PROJECT_NAME}
)

# command run
add_custom_target(run
    COMMAND ${BIN}/${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)


message(STATUS ${CUDA_NVCC_FLAGS})
