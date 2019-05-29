cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(exchange)

add_subdirectory(mapper)
add_subdirectory(optimizer)

add_executable(${PROJECT_NAME} exchange.cpp logger.cpp logger.h)

set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_link_libraries(${PROJECT_NAME}
                      "${TORCH_LIBRARIES}"
                      stdc++fs
                      optimizer
                      mapper)