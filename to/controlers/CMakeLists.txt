cmake_minimum_required(VERSION 3.13)

project(controlers
        VERSION 0.1.0
        LANGUAGES CXX)

include(GNUInstallDirs)

set(LIB_SRC
        collision_alert.cpp
        create_trajectory.cpp
        nearest_neighbour.cpp
        road_actions.cpp
        road_safety.cpp)

set(LIB_H
        include/collision_alert.h
        include/create_trajectory.h
        include/nearest_neighbour.h
        include/road_actions.h
        include/road_safety.h)


add_library(${PROJECT_NAME} SHARED ${LIB_SRC} ${LIB_H})

target_include_directories(includes INTERFACE include)

target_link_libraries(${PROJECT_NAME}
                        "${TORCH_LIBRARIES}"
                        includes)

set_target_properties(${PROJECT_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
