cmake_minimum_required(VERSION 2.6)
project (bag2events)

find_package(OpenCV REQUIRED)
find_package(rosbag REQUIRED)
find_package(catkin_simple REQUIRED)
catkin_simple()
catkin_package()

include_directories(
	include
   	${catkin_INCLUDE_DIRS}
    ${rosbag_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

cs_add_executable(bag2events bag2events.cpp)

target_link_libraries(bag2events ${rosbag_LIBRARIES} ${OpenCV_LIBRARIES})