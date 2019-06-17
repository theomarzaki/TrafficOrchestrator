// New File Made to represent the messages the car/server would receive

// Created by : KCL
#ifndef TO_COLLISION_ALERT_H
#define TO_COLLISION_ALERT_H

#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <tuple>
#include "road_user.h"

using std::string;
using std::vector;
using std::ostream;
using std::tuple;

typedef uint32_t uint4;

class CollisionAlert {

private:

	string type;
	string context;
	string origin;
	tuple<uint8_t, uint8_t, uint8_t> version;
	uint64_t timestamp;
	string uuid_vehicle; // Digital Identifier of RU that shall execute the maneuver.
	string uuid_to; // Digital Identifier of the traffic orchestrator.

  string station_type_ru;// Road User station type
  uint32_t latitude_ru;
  uint32_t longitude_ru;
  uint16_t speed_ru;

	uint32_t latitude_collision;
	uint32_t longitude_collision;

	string signature;

public:

	CollisionAlert(tuple<uint8_t, uint8_t, uint8_t> version, uint64_t timestamp, string uuid_vehicle, string uuid_to,
	uint32_t latitude_collision, uint32_t longitude_collision,string station_type_ru,
  uint32_t latitude_ru,uint32_t longitude_ru,uint32_t speed_ru,string signature) :
	version(version),
	timestamp(timestamp),
	uuid_vehicle(uuid_vehicle),
	uuid_to(uuid_to),
  station_type_ru(station_type_ru),
  latitude_ru(latitude_ru),
  longitude_ru(longitude_ru),
  speed_ru(speed_ru),
	latitude_collision(latitude_collision),
	longitude_collision(longitude_collision),
	signature(signature)
	{
		type = "collision alert";
		context = "lane_merge";
		origin = "traffic_orchestrator";
	}

	CollisionAlert() {
		type = "maneuver";
		context = "lane_merge";
		origin = "traffic_orchestrator";
	}

    friend std::ostream& operator<< (ostream& os, CollisionAlert * collisionRec); // Overload << to print a recommendation.

    string getType();
    string getContext();
    string getOrigin();
    tuple<uint8_t, uint8_t, uint8_t> getVersion();
    uint64_t getTimestamp();
    string getUuidVehicle();
    string getUuidTo();
    string getStationType();
    uint32_t getLatitudeRU();
    uint32_t getLongitudeRU();
    uint16_t getSpeedRU();
    uint32_t getLatitudeCollision();
    uint32_t getLongitudeCollision();
    string getSignature();

    void setType(string);
    void setContext(string);
    void setOrigin(string);
    void setVersion(tuple<uint8_t, uint8_t, uint8_t>);
    void setTimestamp(uint64_t);
    void setUuidVehicle(string);
    void setUuidTo(string);
    void setStationType(string);
    void setLatitudeRU(uint32_t);
    void setLongitudeRU(uint32_t);
    void setSpeedRU(uint32_t);
    void getLatitudeCollision(uint32_t);
    void getLongitudeCollision(uint32_t);
    void setSignature(string);

    ~CollisionAlert();

};

#endif