// This script handles the road user information of the cars in a class structure

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#ifndef TO_ROAD_USER_H
#define TO_ROAD_USER_H

#include <iostream>
#include <ostream>
#include <string>
#include <inttypes.h>
#include <time.h>
#include <math.h>

using std::string;
using std::ostream;

typedef uint32_t uint4;

class RoadUser {

private:

string type;
string context; // Used to distinguish different messages i.e. lane_merge
string origin; // Indicates the creator of the RU description i.e. "self" or "roadside_camera".
string version; // Indicates the version of the message format specification.
uint64_t timestamp; // Indicates when the message information was assembled.

string uuid; // Digital identifier for the RU.
string its_station_type; // Description of DE_StationType from [ETSI14-1028942]. i.e. "bus".

/* Can take value TRUE or FALSE. */
bool connected; // Whether the vehicle can communicate with the V2X gateway.

string position_type;

/* Position with precision to 0.1 microdegrees. */
int32_t latitude; // Specifies the north-south position on the Earth's surface.
int32_t longitude; // Specifies the east-west position on the Earth's surface.

// string position_type TODO

/* Heading with precision to 0.1 degrees from North. */
uint16_t heading; // Driving direction, clockwise measurement.

/* Speed with precision to 0.01 m/s. */
uint16_t speed; // Speed of the RU in direction specified in "heading" field.

/* Acceleration with precision to 0.1 m/(s^2). */
uint16_t acceleration; // Acceleration of the RU in the direction specified in "heading" field.

/* Yaw rate with precision 0.1 degrees/s. */
uint16_t yaw_rate; // Heading change of the RU, where the sign indicates direction. i.e. + or -

/* Size with precision to 0.1m. */
float length; // Length of the RU.
float width; // Width of the RU.
float height; // Height of the RU.

/* Color represented as a hexadecimal RGB value. */
string color; // Color of the vehicle.

uint4 lane_position; // Counted from rightmost lane in driving direction (starting at index 0).

/* Measured in percentage - captures possible object detection issues. */
uint8_t existence_probability; // Indicates the likelihood that the RU exists.

uint16_t position_semi_major_confidence; // Larger eigenvalue of position covariance matrix.
uint16_t position_semi_minor_confidence; // Smaller eigenvalue of position covariance matrix.
uint16_t position_semi_major_orientation; // Direction of eigenvector of position covariance matrix.

/* Heading confidence with precision 0.1 degrees from North. */
uint16_t heading_c; // Variance of heading measurements.

/* Speed confidence with precision 0.01 m/s. */
uint16_t speed_c; // Variance of speed measurements.

/* Acceleration confidence with precision 0.1 m/(s^2). */
uint16_t acceleration_c; // Variance of acceleration measurements.

/* Yaw rate confidence with precision 0.1 degrees/s. */
uint16_t yaw_rate_c; // Variance of yaw rate measurements.

/* Size confidence with precision 0.1m. */
float length_c; // Variance of RU length measurements.
float width_c; // Variance of RU width measurements.
float height_c; // Variance of RU height measurements.

string signature; // Used for signing the message content.
string source_uuid;

/* flag indicating availability of road user to receive waypoint */
bool processing_waypoint;
time_t waypoint_timestamp;

public:

RoadUser(string type,string context,string origin,string version,uint64_t timestamp,string uuid,string its_station_type,
        bool connected,int32_t latitude,int32_t longitude,string position_type,
        uint16_t heading,uint16_t speed,uint16_t acceleration,uint16_t yaw_rate,float length,float width,
        float height,string color,uint4 lane_position,uint8_t existence_probability,uint16_t position_semi_major_confidence,
        uint16_t position_semi_minor_confidence,uint16_t position_semi_major_orientation,uint16_t heading_c,
        uint16_t speed_c,uint16_t acceleration_c,uint16_t yaw_rate_c,float length_c,float width_c,float height_c,
        string signature, string source_uuid) :
type(std::move(type)),
context(std::move(context)),
origin(std::move(origin)),
version(std::move(version)),
timestamp(timestamp),
uuid(std::move(uuid)),
its_station_type(std::move(its_station_type)),
connected(connected),
latitude(latitude),
longitude(longitude),
position_type(std::move(position_type)),
heading(heading),
speed(speed),
acceleration(acceleration),
yaw_rate(yaw_rate),
length(length),
width(width),
height(height),
color(std::move(color)),
lane_position(lane_position),
existence_probability(existence_probability),
position_semi_major_confidence(position_semi_major_confidence),
position_semi_minor_confidence(position_semi_minor_confidence),
position_semi_major_orientation(position_semi_major_orientation),
heading_c(heading_c),
speed_c(speed_c),
acceleration_c(acceleration_c),
yaw_rate_c(yaw_rate_c),
length_c(length_c),
width_c(width_c),
height_c(height_c),
signature(std::move(signature)),
source_uuid(std::move(source_uuid)),
processing_waypoint{false},
waypoint_timestamp{0}
{}

/* Default constructor assigns relevant fields to 'No Value Assigned' if not all values are handed to the above constructor. */
RoadUser() {
type = "ru_description";
context = "lane_merge";
origin = "self";
version = "1.0.0";
processing_waypoint = false;
waypoint_timestamp = 0;
connected = false;
}

~RoadUser(); // Destructor declaration.

friend std::ostream& operator<< (ostream& os, RoadUser * roadUser); // Overload << to display a RoadUser.

string getType();
string getContext();
string getOrigin();
string getVersion();
uint64_t getTimestamp();
string getUuid();
string getItsStationType();
bool getConnected();
int32_t getLatitude();
int32_t getLongitude();
uint16_t getHeading();
uint16_t getSpeed();
uint16_t getAcceleration();
uint16_t getYawRate();
float getLength();
float getWidth();
float getHeight();
string getColor();
uint4 getLanePosition();
uint8_t getExistenceProbability();
uint16_t getPositionSemiMajorConfidence();
uint16_t getPositionSemiMinorConfidence();
uint16_t getPositionSemiMajorOrientation();
uint16_t getHeadingConfidence();
uint16_t getSpeedConfidence();
uint16_t getAccelerationConfidence();
uint16_t getYawRateConfidence();
float getLengthConfidence();
float getWidthConfidence();
float getHeightConfidence();
string getSignature();
string getPositionType();
string getSouceUUID();
bool getProcessingWaypoint();
time_t getWaypointTimestamp();

void setType(string);
void setContext(string);
void setOrigin(string);
void setVersion(string);
void setTimestamp(uint64_t);
void setUuid(string);
void setItsStationType(string);
void setConnected(bool);
void setLatitude(int32_t);
void setLongitude(int32_t);
void setHeading(uint16_t);
void setSpeed(uint16_t);
void setAcceleration(uint16_t);
void setYawRate(uint16_t);
void setLength(float);
void setWidth(float);
void setHeight(float);
void setColor(string);
void setLanePosition(uint4);
void setExistenceProbability(uint8_t);
void setPositionSemiMajorConfidence(uint16_t);
void setPositionSemiMinorConfidence(uint16_t);
void setPositionSemiMajorOrientation(uint16_t);
void setHeadingConfidence(uint16_t);
void setSpeedConfidence(uint16_t);
void setAccelerationConfidence(uint16_t);
void setYawRateConfidence(uint16_t);
void setLengthConfidence(float);
void setWidthConfidence(float);
void setHeightConfidence(float);
void setSignature(string);
void setPositionType(string);
void setSourceUUID(string);
void setProcessingWaypoint(bool);
void setWaypointTimeStamp(time_t);

};

#endif
