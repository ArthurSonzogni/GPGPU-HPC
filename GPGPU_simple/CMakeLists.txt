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

# release build type by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release
        RelWithDebInfo MinSizeRel."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# find files
file(GLOB_RECURSE source_files ${SRC}/*)
file(GLOB_RECURSE common_files ${COM}/*)


# Finding libs
find_package(CUDA REQUIRED)

# set some options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -ccbin g++")


cuda_add_executable(
    ${PROJECT_NAME}
    ${source_files}
    ${common_files}
)

# include
include_directories(
    ${SRC}
    ${INC}
    ${COM}
)

CUDA_INCLUDE_DIRECTORIES(
    ${SRC}
    ${INC}
    ${COM}
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
