cmake_minimum_required(VERSION 3.13)

project(mapper
        VERSION 0.1.0
        LANGUAGES CXX)

include(GNUInstallDirs)

set(LIB_SRC
        mapper.cpp
        ../logger.cpp)

set(LIB_H
        include/mapper.h
        ../logger.h)


add_library(${PROJECT_NAME} STATIC ${LIB_SRC} ${LIB_H})

set_target_properties(${PROJECT_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
