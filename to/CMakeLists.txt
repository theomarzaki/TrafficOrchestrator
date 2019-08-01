cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(exchange)

add_library(includes INTERFACE)

add_subdirectory(optimizer)
add_subdirectory(connectors)
add_subdirectory(controlers)
add_subdirectory(models)
add_subdirectory(mapper)
add_subdirectory(logger)

add_executable(${PROJECT_NAME} exchange.cpp
        models/maneuver_feedback.cpp
        models/maneuver_recommendation.cpp
        models/subscription_request.cpp
        models/subscription_response.cpp
        models/unsubscription_request.cpp
        models/unsubscription_response.cpp
        models/road_user.cpp
        models/database.cpp
        connectors/protocol.cpp
        connectors/network_interface.cpp
        optimizer/optimizer_engine.cpp
        logger/logger.cpp) #cmake is a stupid shit

target_link_libraries(${PROJECT_NAME}
                        stdc++fs
                        controllers
                        models
                        optimizer
                        mapper
                        logger
                        connectors
                        includes)

set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

file(REMOVE ${CMAKE_BINARY_DIR}/logs/Montlhery_dump.json)