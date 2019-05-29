// this script models the way points needed for the vehicle to follow the lane merge algorithm

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#ifndef TO_WAYPOINT_H
#define TO_WAYPOINT_H

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

#endif