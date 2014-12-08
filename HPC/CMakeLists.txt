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

# find files
file(GLOB_RECURSE source_files ${SRC}/*)


# Finding libs
find_package(MPI REQUIRED)

add_executable(
    ${PROJECT_NAME}
    ${source_files}
)

# add OpenMPI flags
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-reorder -Wno-unused-parameter")
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

# include
include_directories(
    ${SRC}
    ${MPI_INCLUDE_PATH}
    ${common_files}
    ${INC}
)

# libs
target_link_libraries(
    ${PROJECT_NAME}
    ${MPI_LIBRARIES}
    ${COM}
)

# command run
add_custom_target(run
    COMMAND ${MPIEXEC} ${MPI_EXEC_NUMPROC_FLAG} ${BIN}/${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
