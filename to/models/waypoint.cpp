// this script models the way points needed for the vehicle to follow the lane merge algorithm

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#include "include/waypoint.h"

Waypoint::~Waypoint() = default;

uint64_t Waypoint::getTimestamp() {return timestamp;}
uint32_t Waypoint::getLatitude() {return latitude;}
uint32_t Waypoint::getLongitude() {return longitude;}
uint16_t Waypoint::getSpeed() {return speed;}
uint4 Waypoint::getLanePosition() {return lane_position;}
uint16_t Waypoint::getHeading() {return heading;}

void Waypoint::setTimestamp(uint64_t parameter){timestamp = parameter;}
void Waypoint::setLatitude(uint32_t parameter){latitude = parameter;}
void Waypoint::setLongitude(uint32_t parameter){longitude = parameter;}
void Waypoint::setSpeed(uint16_t parameter){speed = parameter;}
void Waypoint::setLanePosition(uint4 parameter){lane_position = parameter;}
void Waypoint::setHeading(uint16_t parameter){heading = parameter;}


std::ostream& operator<<(std::ostream& os, Waypoint * waypoint) {

  os
  << "["
  << waypoint->getTimestamp()
  << ","
  << waypoint->getLatitude()
  << ","
  << waypoint->getLongitude()
  << ","
  << waypoint->getSpeed()
  << ","
  << waypoint->getLanePosition()
  << "]\n";
  return os;

}
