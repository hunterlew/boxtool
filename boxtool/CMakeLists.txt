cmake_minimum_required(VERSION 3.5)
add_definitions(-std=c++11)

project(boxtool)

find_package(OpenCV 2.4.13 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(
    box_rotation
    box_rotation.cpp
)

target_link_libraries(
    box_rotation
    ${OpenCV_LIBS}
)
