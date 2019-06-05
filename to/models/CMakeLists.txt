cmake_minimum_required(VERSION 3.13)

project(models
        VERSION 0.1.0
        LANGUAGES CXX)

include(GNUInstallDirs)

set(LIB_SRC
        database.cpp
        maneuver_feedback.cpp
        maneuver_recommendation.cpp
        road_user.cpp
        subscription_request.cpp
        subscription_response.cpp
        unsubscription_request.cpp
        unsubscription_response.cpp
        waypoint.cpp)

set(LIB_H
        include/database.h
        include/maneuver_feedback.h
        include/maneuver_recommendation.h
        include/road_user.h
        include/subscription_request.h
        include/subscription_response.h
        include/unsubscription_request.h
        include/unsubscription_response.h
        include/waypoint.h)

add_library(${PROJECT_NAME} SHARED ${LIB_SRC} ${LIB_H})

target_include_directories(includes INTERFACE include)

target_link_libraries(${PROJECT_NAME} includes)

set_target_properties(${PROJECT_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
