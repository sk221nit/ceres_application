cmake_minimum_required(VERSION 2.8.3)

project(ceres_application)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -Wall -g")

find_package(OpenCV)

find_package(Ceres REQUIRED)
find_package(Eigen3)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
    src/ceres_base.cc
)


target_link_libraries(${PROJECT_NAME} ${EIGEN3_LIBRARIES}  ${OpenCV_LIBS} ${CERES_LIBRARIES}) 

add_executable(CERES_APP
src/ceres_app.cc)
target_link_libraries(CERES_APP ${PROJECT_NAME})