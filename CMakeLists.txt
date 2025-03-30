cmake_minimum_required(VERSION 3.0.2)
project(align_coordinate)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-w")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -ggdb ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g ${CMAKE_CXX_FLAGS}")


find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        nav_msgs
        sensor_msgs
        roscpp
        rospy
        std_msgs
        pcl_ros
        tf2_ros
        eigen_conversions
        )


find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)

# sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

catkin_package(
        CATKIN_DEPENDS
        geometry_msgs
        nav_msgs
        roscpp
        rospy
        std_msgs

        DEPENDS EIGEN3 PCL
        INCLUDE_DIRS
)


include_directories(
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        include
)

add_executable(align_coordinate src/align_coordinate.cpp)
target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        tbb
        glog
        )
