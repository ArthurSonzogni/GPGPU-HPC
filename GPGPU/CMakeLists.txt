cmake_minimum_required(VERSION 2.6)

# project name
set(PROJECT_NAME GPGPU)
project(${PROJECT_NAME})

# bin and src dir
set(BIN ${PROJECT_SOURCE_DIR}/bin)
set(SRC ${PROJECT_SOURCE_DIR}/src)


# find files
file(GLOB_RECURSE source_files ${SRC}/*)


# Finding libs
find_package(CUDA REQUIRED)

cuda_add_executable(
    ${PROJECT_NAME}
    ${source_files}
)

# include
include_directories(
    ${SRC}
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

# set some options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-reorder -Wno-unused-parameter")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_22,code=sm_22")
