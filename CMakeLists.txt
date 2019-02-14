cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(communication_layer)

set( PROJECT communication_layer )
message( ${CMAKE_CURRENT_SOURCE_DIR} )

include_directories("${CMAKE_SOURCE_DIR}/rapidjson/"
    "${CMAKE_SOURCE_DIR}/")

include_directories(${communication_layer})
find_package(Torch REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
add_executable(exchange to/exchange.cpp)
target_link_libraries(exchange "${TORCH_LIBRARIES}")
set_property(TARGET exchange PROPERTY CXX_STANDARD 11)

target_include_directories(exchange PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
