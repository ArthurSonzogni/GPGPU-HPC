cmake_minimum_required(VERSION 2.6)

# project name
set(PROJECT_NAME Sequential)
project(${PROJECT_NAME})

# bin,src,common and include dir
set(BIN ${PROJECT_SOURCE_DIR}/bin)
set(SRC ${PROJECT_SOURCE_DIR}/src)
set(COM ${PROJECT_SOURCE_DIR}/../Common/src)
set(INC ${PROJECT_SOURCE_DIR}/../Common/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-reorder -Wno-unused-parameter")

# release build type by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release
        RelWithDebInfo MinSizeRel."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# output dir under ./bin
set(EXECUTABLE_OUTPUT_PATH ${BIN})

# find files
file(GLOB_RECURSE common_files ${COM}/*)
file(GLOB_RECURSE source_files ${SRC}/*)


add_executable(
    ${PROJECT_NAME}
    ${source_files}
    ${common_files}
)


# include
include_directories(
    ${SRC}
    ${COM}
    ${INC}
)

# link
target_link_libraries(
    ${PROJECT_NAME}
    ${COM}
)


# command run
add_custom_target(run
    COMMAND ${BIN}/${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# command write
add_custom_target(write
    COMMAND ${BIN}/${PROJECT_NAME} -write 1
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
