// New File Made to represent the messages the car/server would receive

// Created by : KCL
#include "collision_alert.h"

CollisionAlert::CollisionAlert(std::tuple<uint8_t, uint8_t, uint8_t> version, uint64_t timestamp, std::string uuid_vehicle, std::string uuid_to,
                               uint32_t latitude_collision, uint32_t longitude_collision,std::string station_type_ru,
                               uint32_t latitude_ru,uint32_t longitude_ru,uint32_t speed_ru,std::string signature) :
    type("collision alert"),
    version(std::move(version)),
    timestamp(timestamp),
    uuid_vehicle(std::move(uuid_vehicle)),
    uuid_to(std::move(uuid_to)),
    station_type_ru(std::move(station_type_ru)),
    latitude_ru(latitude_ru),
    longitude_ru(longitude_ru),
    speed_ru(speed_ru),
    latitude_collision(latitude_collision),
    longitude_collision(longitude_collision),
    signature(std::move(signature))
{}

std::string CollisionAlert::getType() {return type;}
std::string CollisionAlert::getContext(){return context;}
std::string CollisionAlert::getOrigin(){return origin;}
std::tuple<uint8_t, uint8_t, uint8_t> CollisionAlert::getVersion(){return version;}
std::uint64_t CollisionAlert::getTimestamp(){return timestamp;}
std::string CollisionAlert::getUuidVehicle(){return uuid_vehicle;}
std::string CollisionAlert::getUuidTo(){return uuid_to;}
std::string CollisionAlert::getStationType(){return station_type_ru;}
uint32_t CollisionAlert::getLatitudeRU(){return latitude_ru;}
uint32_t CollisionAlert::getLongitudeRU(){return longitude_ru;}
uint16_t CollisionAlert::getSpeedRU(){return speed_ru;}
uint32_t CollisionAlert::getLatitudeCollision(){return latitude_collision;}
uint32_t CollisionAlert::getLongitudeCollision(){return longitude_collision;}

std::string CollisionAlert::getSignature(){return signature;}

void CollisionAlert::setType(std::string parameter){type = std::move(parameter);}
void CollisionAlert::setContext(std::string parameter){context = std::move(parameter);}
void CollisionAlert::setOrigin(std::string parameter){origin = std::move(parameter);}
void CollisionAlert::setVersion(std::tuple<uint8_t, uint8_t, uint8_t> parameter){version = parameter;}
void CollisionAlert::setTimestamp(uint64_t parameter){timestamp = parameter;}
void CollisionAlert::setUuidVehicle(std::string parameter){uuid_vehicle = std::move(parameter);}
void CollisionAlert::setUuidTo(std::string parameter){uuid_to = std::move(parameter);}
void CollisionAlert::setStationType(std::string parameter){station_type_ru = std::move(parameter);}
void CollisionAlert::setLatitudeRU(uint32_t parameter){latitude_ru = parameter;}
void CollisionAlert::setLongitudeRU(uint32_t parameter){longitude_ru = parameter;}
void CollisionAlert::setSpeedRU(uint32_t parameter){speed_ru = parameter;}
void CollisionAlert::getLatitudeCollision(uint32_t parameter){latitude_collision = parameter;}
void CollisionAlert::getLongitudeCollision(uint32_t parameter){longitude_collision = parameter;}
void CollisionAlert::setSignature(std::string parameter){signature = std::move(parameter);}

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
