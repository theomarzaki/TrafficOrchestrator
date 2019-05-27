cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(communication_layer)

add_subdirectory(thirdparty)

include_directories(${CMAKE_SOURCE_DIR}/rapidjson
        ${CMAKE_SOURCE_DIR}
        ${communication_layer}
        ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/libtorch)
find_package(Torch REQUIRED)

add_executable(exchange
        to/exchange.cpp
        to/logger.cpp)

target_link_libraries(exchange "${TORCH_LIBRARIES}" stdc++fs)

target_include_directories(exchange PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# install project config and version file
file(COPY include DESTINATION ${CMAKE_BINARY_DIR})