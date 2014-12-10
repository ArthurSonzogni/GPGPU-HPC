cmake_minimum_required(VERSION 2.6)

# project name
set(PROJECT_NAME HPC)
project(${PROJECT_NAME})

# bin,src,common and include dir
set(BIN ${PROJECT_SOURCE_DIR}/bin)
set(SRC ${PROJECT_SOURCE_DIR}/src)
set(COM ${PROJECT_SOURCE_DIR}/../Common/src)
set(INC ${PROJECT_SOURCE_DIR}/../Common/include)

# output dir under ./bin
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

find_package(OpenMP REQUIRED)

add_executable(
    ${PROJECT_NAME}
    ${source_files}
    ${common_files}
)



# add OpenMP flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-reorder -Wno-unused-parameter ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-reorder -Wno-unused-parameter ${OpenMP_C_FLAGS}")

# include
include_directories(
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
