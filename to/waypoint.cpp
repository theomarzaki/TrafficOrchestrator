// this script models the way points needed for the vehicle to follow the lane merge algorithm

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#include <iostream>
#include <ostream>

using namespace std;

typedef uint32_t uint4;

class Waypoint {

private:

	uint64_t timestamp;
	uint32_t latitude;
	uint32_t longitude;
	uint16_t speed;
	uint4 lane_position;
	uint16_t heading;

public:

	Waypoint(uint64_t timestamp, uint32_t latitude, uint32_t longitude, uint16_t speed, uint4 lane_position) :
	timestamp(timestamp),
	latitude(latitude),
	longitude(longitude),
	speed(speed),
	lane_position(lane_position)
	{}

	Waypoint() {}

	~Waypoint();

	friend ostream& operator<<(ostream& os, Waypoint * waypoint); // Overload << to print out a Waypoint pointer.

	uint64_t getTimestamp();
	uint32_t getLatitude();
	uint32_t getLongitude();
	uint16_t getSpeed();
	uint4 getLanePosition();
	uint16_t getHeading();

	void setTimestamp(uint64_t);
	void setLatitude(uint32_t);
	void setLongitude(uint32_t);
	void setSpeed(uint16_t);
	void setLanePosition(uint4);
	void setHeading(uint16_t);

};

Waypoint::~Waypoint() {}

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


ostream& operator<<(ostream& os, Waypoint * waypoint) {

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
