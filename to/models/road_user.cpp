// This script handles the road user information of the cars in a class structure

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#include "include/road_user.h"

RoadUser::RoadUser()  {
    type = "ru_description";
    context = "lane_merge";
    origin = "self";
    version = "1.0.0";
    processing_waypoint = false;
    waypoint_timestamp = 0;
}

std::string RoadUser::getType(){return type;}
std::string RoadUser::getContext(){return context;}
std::string RoadUser::getOrigin(){return origin;}
std::string RoadUser::getVersion(){return version;}
uint64_t RoadUser::getTimestamp(){return timestamp;}
std::string RoadUser::getUuid(){return uuid;}
std::string RoadUser::getItsStationType(){return its_station_type;}
bool RoadUser::getConnected(){return connected;}
int32_t RoadUser::getLatitude(){return latitude;}
int32_t RoadUser::getLongitude(){return longitude;}
uint16_t RoadUser::getHeading(){return heading;}
uint16_t RoadUser::getSpeed(){return speed;}
uint16_t RoadUser::getAcceleration(){return acceleration;}
uint16_t RoadUser::getYawRate(){return yaw_rate;}
float RoadUser::getLength(){return length;}
float RoadUser::getWidth(){return width;}
float RoadUser::getHeight(){return height;}
std::string RoadUser::getColor(){return color;}
uint4 RoadUser::getLanePosition(){return lane_position;}
uint8_t RoadUser::getExistenceProbability(){return existence_probability;}
uint16_t RoadUser::getPositionSemiMajorConfidence(){return position_semi_major_confidence;}
uint16_t RoadUser::getPositionSemiMinorConfidence(){return position_semi_minor_confidence;}
uint16_t RoadUser::getPositionSemiMajorOrientation(){return position_semi_major_orientation;}
uint16_t RoadUser::getHeadingConfidence(){return heading_c;}
uint16_t RoadUser::getSpeedConfidence(){return speed_c;}
uint16_t RoadUser::getAccelerationConfidence(){return acceleration_c;}
uint16_t RoadUser::getYawRateConfidence(){return yaw_rate_c;}
float RoadUser::getLengthConfidence(){return length_c;}
float RoadUser::getWidthConfidence() {return width_c;}
float RoadUser::getHeightConfidence(){return height_c;}
std::string RoadUser::getSignature(){return signature;}
std::string RoadUser::getPositionType(){return position_type;}
std::string RoadUser::getSouceUUID(){return source_uuid;}
bool RoadUser::getProcessingWaypoint(){return processing_waypoint;}
time_t RoadUser::getWaypointTimestamp(){return waypoint_timestamp;}

void RoadUser::setType(std::string parameter){type = std::move(parameter);}
void RoadUser::setContext(std::string parameter){context = std::move(parameter);}
void RoadUser::setOrigin(std::string parameter){origin = std::move(parameter);}
void RoadUser::setVersion(std::string parameter){version = std::move(parameter);}
void RoadUser::setTimestamp(uint64_t parameter){timestamp = parameter;}
void RoadUser::setUuid(std::string parameter){uuid = std::move(parameter);}
void RoadUser::setItsStationType(std::string parameter){its_station_type = std::move(parameter);}
void RoadUser::setConnected(bool parameter){connected = parameter;}
void RoadUser::setLatitude(int32_t parameter){latitude = parameter;}
void RoadUser::setLongitude(int32_t parameter){longitude = parameter;}
void RoadUser::setHeading(uint16_t parameter){heading = parameter;}
void RoadUser::setSpeed(uint16_t parameter){speed = parameter;}
void RoadUser::setAcceleration(uint16_t parameter){acceleration = parameter;}
void RoadUser::setYawRate(uint16_t parameter){yaw_rate = parameter;}
void RoadUser::setLength(float parameter){length = parameter;}
void RoadUser::setWidth(float parameter){width = parameter;}
void RoadUser::setHeight(float parameter){height = parameter;}
void RoadUser::setColor(std::string parameter){color = std::move(parameter);}
void RoadUser::setLanePosition(uint4 parameter){lane_position = parameter;}
void RoadUser::setExistenceProbability(uint8_t parameter){existence_probability = parameter;}
void RoadUser::setPositionSemiMajorConfidence(uint16_t parameter){position_semi_major_confidence = parameter;}
void RoadUser::setPositionSemiMinorConfidence(uint16_t parameter){position_semi_minor_confidence = parameter;}
void RoadUser::setPositionSemiMajorOrientation(uint16_t parameter){position_semi_major_orientation = parameter;}
void RoadUser::setHeadingConfidence(uint16_t parameter){heading_c = parameter;}
void RoadUser::setSpeedConfidence(uint16_t parameter){speed_c = parameter;}
void RoadUser::setAccelerationConfidence(uint16_t parameter){acceleration_c = parameter;}
void RoadUser::setYawRateConfidence(uint16_t parameter){yaw_rate_c = parameter;}
void RoadUser::setLengthConfidence(float parameter){yaw_rate_c = static_cast<uint16_t>(parameter);}
void RoadUser::setWidthConfidence(float parameter) {width_c = parameter;}
void RoadUser::setHeightConfidence(float parameter){height_c = parameter;}
void RoadUser::setSignature(std::string parameter){signature = std::move(parameter);}
void RoadUser::setPositionType(std::string parameter){position_type = std::move(parameter);}
void RoadUser::setSourceUUID(std::string parameter){source_uuid = std::move(parameter);}
void RoadUser::setProcessingWaypoint(bool parameter){processing_waypoint = parameter;}
void RoadUser::setWaypointTimeStamp(time_t parameter){waypoint_timestamp = parameter;}

std::ostream& operator<<(std::ostream& os, RoadUser * roadUser) {
  os
  << "["
  << roadUser->getType()
  << ","
  << roadUser->getContext()
  << ","
  << roadUser->getOrigin()
  << ","
  << roadUser->getVersion()
  << ","
  << roadUser->getTimestamp()
  << ","
  << roadUser->getUuid()
  << ","
  << roadUser->getItsStationType()
  << ","
  << roadUser->getConnected()
  << ","
  << roadUser->getLatitude()
  << ","
  << roadUser->getLongitude()
  << ","
  << roadUser->getPositionType()
  << ","
  << roadUser->getSouceUUID()
  << ","
  << roadUser->getHeading()
  << ","
  << roadUser->getSpeed()
  << ","
  << roadUser->getAcceleration()
  << ","
  << roadUser->getYawRate()
  << ","
  << roadUser->getLength()
  << ","
  << roadUser->getWidth()
  << ","
  << roadUser->getHeight()
  << ","
  << roadUser->getColor()
  << ","
  << roadUser->getLanePosition()
  << ","
  << +roadUser->getExistenceProbability()
  << ","
  << roadUser->getPositionSemiMajorConfidence()
  << ","
  << roadUser->getPositionSemiMinorConfidence()
  << ","
  << roadUser->getPositionSemiMajorOrientation()
  << ","
  << roadUser->getHeadingConfidence()
  << ","
  << roadUser->getSpeedConfidence()
  << ","
  << roadUser->getAccelerationConfidence()
  << ","
  << roadUser->getYawRateConfidence()
  << ","
  << roadUser->getLengthConfidence()
  << ","
  << roadUser->getWidthConfidence()
  << ","
  << roadUser->getHeightConfidence()
  << ","
  << roadUser->getSignature()
  << ","
  << roadUser->getProcessingWaypoint()
  << "]\n";

  return os;
}
