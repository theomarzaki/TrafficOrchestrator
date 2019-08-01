// New File Made to represent the messages the car/server would receive

// Created by : KCL
#ifndef TO_COLLISION_ALERT_H
#define TO_COLLISION_ALERT_H

#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <tuple>
#include <utility>
#include "road_user.h"

typedef uint32_t uint4;

class CollisionAlert {

private:

    std::string type{"maneuver"};
    std::string context{"lane_merge"};
    std::string origin{"traffic_orchestrator"};
    std::tuple<uint8_t, uint8_t, uint8_t> version{0,0,0};
    uint64_t timestamp{0};
    std::string uuid_vehicle{""}; // Digital Identifier of RU that shall execute the maneuver.
    std::string uuid_to{""}; // Digital Identifier of the traffic orchestrator.
    std::string station_type_ru{""};// Road User station type
    uint32_t latitude_ru{0};
    uint32_t longitude_ru{0};
    uint16_t speed_ru{0};
    uint32_t latitude_collision{0};
    uint32_t longitude_collision{0};
    std::string signature{""};

public:

	CollisionAlert( std::tuple<uint8_t, uint8_t, uint8_t> version, uint64_t timestamp, std::string uuid_vehicle, std::string uuid_to,
                    uint32_t latitude_collision, uint32_t longitude_collision,std::string station_type_ru,
                    uint32_t latitude_ru,uint32_t longitude_ru,uint32_t speed_ru,std::string signature);
    CollisionAlert() = default;

    ~CollisionAlert() = default;

    friend std::ostream& operator<< (ostream& os, CollisionAlert * collisionRec); // Overload << to print a recommendation.

    std::string getType();
    std::string getContext();
    std::string getOrigin();
    std::tuple<uint8_t, uint8_t, uint8_t> getVersion();
    uint64_t getTimestamp();
    std::string getUuidVehicle();
    std::string getUuidTo();
    std::string getStationType();
    uint32_t getLatitudeRU();
    uint32_t getLongitudeRU();
    uint16_t getSpeedRU();
    uint32_t getLatitudeCollision();
    uint32_t getLongitudeCollision();
    std::string getSignature();

    void setType(std::string);
    void setContext(std::string);
    void setOrigin(std::string);
    void setVersion(std::tuple<uint8_t, uint8_t, uint8_t>);
    void setTimestamp(uint64_t);
    void setUuidVehicle(std::string);
    void setUuidTo(std::string);
    void setStationType(std::string);
    void setLatitudeRU(uint32_t);
    void setLongitudeRU(uint32_t);
    void setSpeedRU(uint32_t);
    void getLatitudeCollision(uint32_t);
    void getLongitudeCollision(uint32_t);
    void setSignature(std::string);

};

#endif