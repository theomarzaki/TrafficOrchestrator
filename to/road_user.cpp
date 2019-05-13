// This script handles the road user information of the cars in a class structure

// Created by: KCL

// Modified by: Omar Nassef (KCL)


#include <iostream>
#include <ostream>
#include <string>
#include <inttypes.h>
#include <time.h>

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

/* flag indicating availability of road user to recieve waypoint */
bool processing_waypoint;
time_t waypoint_timestamp;

public:

RoadUser(string type,string context,string origin,string version,uint64_t timestamp,string uuid,string its_station_type,bool connected,int32_t latitude,int32_t longitude,string position_type,
uint16_t heading,uint16_t speed,uint16_t acceleration,uint16_t yaw_rate,float length,float width,
float height,string color,uint4 lane_position,uint8_t existence_probability,uint16_t position_semi_major_confidence,
uint16_t position_semi_minor_confidence,uint16_t position_semi_major_orientation,uint16_t heading_c,
uint16_t speed_c,uint16_t acceleration_c,uint16_t yaw_rate_c,float length_c,float width_c,float height_c,string signature, string source_uuid) :
type(type),
context(context),
origin(origin),
version(version),
timestamp(timestamp),
uuid(uuid),
its_station_type(its_station_type),
connected(connected),
latitude(latitude),
longitude(longitude),
position_type(position_type),
heading(heading),
speed(speed),
acceleration(acceleration),
yaw_rate(yaw_rate),
length(length),
width(width),
height(height),
color(color),
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
signature(signature),
source_uuid(source_uuid)
{}

/* Default constructor assigns relevant fields to 'No Value Assigned' if not all values are handed to the above constructor. */
RoadUser() {
type = "ru_description";
context = "lane_merge";
origin = "self";
version = "1.0.0";
processing_waypoint = false;
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
double getDoubleLatitude();
double getDoubleLongitude();
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
void setDoubleLatitude(double);
void setDoubleLongitude(double);
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

RoadUser::~RoadUser(){}

string RoadUser::getType(){return type;}
string RoadUser::getContext(){return context;}
string RoadUser::getOrigin(){return origin;}
string RoadUser::getVersion(){return version;}
uint64_t RoadUser::getTimestamp(){return timestamp;}
string RoadUser::getUuid(){return uuid;}
string RoadUser::getItsStationType(){return its_station_type;}
bool RoadUser::getConnected(){return connected;}
int32_t RoadUser::getLatitude(){return latitude;}
int32_t RoadUser::getLongitude(){return longitude;}
double RoadUser::getDoubleLatitude(){return latitude/1000000.0;}
double RoadUser::getDoubleLongitude(){return longitude/1000000.0;}
uint16_t RoadUser::getHeading(){return heading;}
uint16_t RoadUser::getSpeed(){return speed;}
uint16_t RoadUser::getAcceleration(){return acceleration;}
uint16_t RoadUser::getYawRate(){return yaw_rate;}
float RoadUser::getLength(){return length;}
float RoadUser::getWidth(){return width;}
float RoadUser::getHeight(){return height;}
string RoadUser::getColor(){return color;}
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
string RoadUser::getSignature(){return signature;}
string RoadUser::getPositionType(){return position_type;}
string RoadUser::getSouceUUID(){return source_uuid;}
bool RoadUser::getProcessingWaypoint(){return processing_waypoint;}
time_t RoadUser::getWaypointTimestamp(){return waypoint_timestamp;}

void RoadUser::setType(string parameter){type = std::move(parameter);}
void RoadUser::setContext(string parameter){context = std::move(parameter);}
void RoadUser::setOrigin(string parameter){origin = std::move(parameter);}
void RoadUser::setVersion(string parameter){version = std::move(parameter);}
void RoadUser::setTimestamp(uint64_t parameter){timestamp = parameter;}
void RoadUser::setUuid(string parameter){uuid = std::move(parameter);}
void RoadUser::setItsStationType(string parameter){its_station_type = std::move(parameter);}
void RoadUser::setConnected(bool parameter){connected = parameter;}
void RoadUser::setLatitude(int32_t parameter){latitude = parameter;}
void RoadUser::setLongitude(int32_t parameter){longitude = parameter;}
void RoadUser::setDoubleLatitude(double parameter){latitude = static_cast<int32_t >(parameter * 1000000);}
void RoadUser::setDoubleLongitude(double parameter){longitude = static_cast<int32_t >(parameter * 1000000);}
void RoadUser::setHeading(uint16_t parameter){heading = parameter;}
void RoadUser::setSpeed(uint16_t parameter){speed = parameter;}
void RoadUser::setAcceleration(uint16_t parameter){acceleration = parameter;}
void RoadUser::setYawRate(uint16_t parameter){yaw_rate = parameter;}
void RoadUser::setLength(float parameter){length = parameter;}
void RoadUser::setWidth(float parameter){width = parameter;}
void RoadUser::setHeight(float parameter){height = parameter;}
void RoadUser::setColor(string parameter){color = std::move(parameter);}
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
void RoadUser::setSignature(string parameter){signature = std::move(parameter);}
void RoadUser::setPositionType(string parameter){position_type = std::move(parameter);}
void RoadUser::setSourceUUID(string parameter){source_uuid = std::move(parameter);}
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
