cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(exchange)

add_subdirectory(mapper)
add_subdirectory(optimizer)
add_subdirectory(logger)

add_executable(${PROJECT_NAME} exchange.cpp logger/logger.cpp logger/include/logger.h) #cmake is a stupid shit

set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_link_libraries(${PROJECT_NAME}
                      "${TORCH_LIBRARIES}"
                      stdc++fs
                      logger
                      optimizer
                      mapper)