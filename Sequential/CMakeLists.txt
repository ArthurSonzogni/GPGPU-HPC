cmake_minimum_required(VERSION 2.6)

# project name
set(PROJECT_NAME Sequential)
project(${PROJECT_NAME})

# bin,src and common dir
set(BIN ${PROJECT_SOURCE_DIR}/bin)
set(SRC ${PROJECT_SOURCE_DIR}/src)
set(COM ${PROJECT_SOURCE_DIR}/../Common/src)


# output dir under ./bin
set(EXECUTABLE_OUTPUT_PATH ${BIN})

# find files
file(GLOB_RECURSE common_files ${COM}/*)
file(GLOB_RECURSE source_files ${SRC}/*)

# Finding libs
find_package(MPI REQUIRED)

add_executable(
    ${PROJECT_NAME}
    ${source_files}
    ${common_files}
)


# include
include_directories(
    ${SRC}
    ${COM}
)


# command run
add_custom_target(run
    COMMAND ${BIN}/${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
