cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(to)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options("-Wall")
add_compile_options("-Werror")
add_compile_options("-Wextra")
add_compile_options("-O0")
add_compile_options("-g")

add_subdirectory(thirdparty)

include_directories(${CMAKE_SOURCE_DIR}/rapidjson
        ${CMAKE_SOURCE_DIR}
        ${communication_layer}
        ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/libtorch)
find_package(Torch REQUIRED)

add_subdirectory(to)

# install project config and version file
file(COPY include DESTINATION ${CMAKE_BINARY_DIR})