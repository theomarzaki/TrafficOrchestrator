cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(exchange)

add_subdirectory(models)
add_subdirectory(mapper)
add_subdirectory(optimizer)
add_subdirectory(logger)
add_subdirectory(connectors)
add_subdirectory(controlers)

add_executable(${PROJECT_NAME} exchange.cpp) #cmake is a stupid shit

set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

target_link_libraries(${PROJECT_NAME} PUBLIC
                      stdc++fs
                      models
                      logger
                      controlers
                      connectors
                      optimizer
                      mapper)