// New File Made to represent the messages the car/server would receive

// Created by : KCL
#include "include/collision_alert.h"

CollisionAlert::~CollisionAlert() = default;

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
uint16_t CollisionAlert::getSpeedRU(){return speed_ru;}
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
void CollisionAlert::setStationType(string parameter){station_type_ru = parameter;}
void CollisionAlert::setLatitudeRU(uint32_t parameter){latitude_ru = parameter;}
void CollisionAlert::setLongitudeRU(uint32_t parameter){longitude_ru = parameter;}
void CollisionAlert::setSpeedRU(uint32_t parameter){speed_ru = parameter;}
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
