// New File Made to represent the messages the car/server would receive

#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <tuple>
#include "road_user.cpp"

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
	type(type),
	context(context),
	origin(origin),
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
void setSpeedRU(uint16_t);
void getLatitudeCollision(uint32_t);
void getLongtitudeCollision(uint32_t);
void setSignature(string);


~CollisionAlert();

};

CollisionAlert::~CollisionAlert() {}

string CollisionAlert::getType() {return type;}
string CollisionAlert::getContext(){return context;}
string CollisionAlert::getOrigin(){return origin;}
tuple<uint8_t, uint8_t, uint8_t> CollisionAlert::getVersion(){return version;}
uint64_t CollisionAlert::getTimestamp(){return timestamp;}
string CollisionAlert::getUuidVehicle(){return uuid_vehicle;}
string CollisionAlert::getUuidTo(){return uuid_to;}
string CollisionAlert::getStationType(){return station_type_ru;}
uint32_t CollisionAlert::getLatitudeRU(){return latitude_ru;}
uint32_t CollisionAlert::getLongitudeRU(){return longitude_ru;}
uint16_t CollisionAlert::getSpeedRU(){return speed_action;}
uint32_t CollisionAlert::getLatitudeCollision(){return latitude_collision;}
uint32_t CollisionAlert::getLongitudeCollision(){return longitude_collision;}

string CollisionAlert::getSignature(){return signature;}

void CollisionAlert::setType(string parameter){type = parameter;}
void CollisionAlert::setContext(string parameter){context = parameter;}
void CollisionAlert::setOrigin(string parameter){origin = parameter;}
void CollisionAlert::setVersion(tuple<uint8_t, uint8_t, uint8_t> parameter){version = parameter;}
void CollisionAlert::setTimestamp(uint64_t parameter){timestamp = parameter;}
void CollisionAlert::setUuidVehicle(string parameter){uuid_vehicle = parameter;}
void CollisionAlert::setUuidTo(string parameter){uuid_to = parameter;}
void CollisionAlert::getStationType(string parameter){station_type_ru = parameter;}
void CollisionAlert::getLatitudeRU(uint32_t parameter){latitude_ru = parameter;}
void CollisionAlert::getLongitudeRU(uint32_t parameter){longitude_ru = parameter;}
void CollisionAlert::getSpeedRU(uint32_t = parameter){speed_action = parameter;}
void CollisionAlert::getLatitudeCollision(uint32_t parameter){latitude_collision = parameter;}
void CollisionAlert::getLongitudeCollision(uint32_t parameter){longitude_collision = parameter;}
void CollisionAlert::setSignature(string parameter){signature = parameter;}

std::ostream& operator<<(std::ostream& os, CollisionAlert * collisionRec) {

  os
  << "["
  << collisionRec->getType()
  << ","
  << collisionRec->getContext()
  << ","
  << collisionRec->getOrigin()
  << ","
  << "[" << std::get<0>(collisionRec->getVersion()) << "," << std::get<1>(collisionRec->getVersion()) << "," << std::get<2>(collisionRec->getVersion()) << "]"
  << ","
  << collisionRec->getTimestamp()
  << ","
  << collisionRec->getUuidVehicle()
  << ","
  << collisionRec->getUuidTo()
  << ","
  << collisionRec->getStationType()
  << ","
  << collisionRec->getLatitudeRU()
  << ","
  << collisionRec->getLongitudeRU()
  << ","
  << collisionRec->getSpeedRU()
  << ","
  << collisionRec->getLongitudeCollision()
  << ","
  << collisionRec->getLongitudeCollision()
  << "]\n";


  return os;

}
