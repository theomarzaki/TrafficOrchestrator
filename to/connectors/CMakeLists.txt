cmake_minimum_required(VERSION 3.13)

project(connectors
        VERSION 0.1.0
        LANGUAGES CXX)

include(GNUInstallDirs)

set(LIB_SRC
        network_interface.cpp)

set(LIB_H
        include/network_interface.h)


add_library(${PROJECT_NAME} SHARED ${LIB_SRC} ${LIB_H})

target_include_directories(includes INTERFACE include)

target_link_libraries(${PROJECT_NAME} includes)

set_target_properties(${PROJECT_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
