// This script handles the road user information of the cars in a class structure

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#ifndef TO_ROAD_USER_H
#define TO_ROAD_USER_H

#include <string>
#include <iostream>
#include <ostream>
#include <inttypes.h>
#include <time.h>
#include <math.h>

using std::ostream;

typedef uint32_t uint4;

class RoadUser {
private:

    std::string type{"ru_description"};
    std::string context{"lane_merge"}; // Used to distinguish different messages i.e. lane_merge
    std::string origin{"self"}; // Indicates the creator of the RU description i.e. "self" or "roadside_camera".
    std::string version{"1.0.0"}; // Indicates the version of the message format specification.
    uint64_t timestamp{0}; // Indicates when the message information was assembled.

    std::string uuid{""}; // Digital identifier for the RU.
    std::string its_station_type{""}; // Description of DE_StationType from [ETSI14-1028942]. i.e. "bus".

    /* Can take value TRUE or FALSE. */
    bool connected{false}; // Whether the vehicle can communicate with the V2X gateway.

    std::string position_type{""};

    /* Position with precision to 0.1 microdegrees. */
    int32_t latitude{0}; // Specifies the north-south position on the Earth's surface.
    int32_t longitude{0}; // Specifies the east-west position on the Earth's surface.

    // std::string position_type TODO

    /* Heading with precision to 0.1 degrees from North. */
    uint16_t heading{0}; // Driving direction, clockwise measurement.

    /* Speed with precision to 0.01 m/s. */
    uint16_t speed{0}; // Speed of the RU in direction specified in "heading" field.

    /* Acceleration with precision to 0.1 m/(s^2). */
    int16_t acceleration{0}; // Acceleration of the RU in the direction specified in "heading" field.

    /* Yaw rate with precision 0.1 degrees/s. */
    int16_t yaw_rate{0}; // Heading change of the RU, where the sign indicates direction. i.e. + or -

    /* Size with precision to 0.1m. */
    float length{0}; // Length of the RU.
    float width{0}; // Width of the RU.
    float height{0}; // Height of the RU.

    /* Color represented as a hexadecimal RGB value. */
    std::string color{""}; // Color of the vehicle.

    uint4 lane_position{0}; // Counted from rightmost lane in driving direction (starting at index 0).

    /* Measured in percentage - captures possible object detection issues. */
    uint8_t existence_probability{0}; // Indicates the likelihood that the RU exists.

    uint16_t position_semi_major_confidence{0}; // Larger eigenvalue of position covariance matrix.
    uint16_t position_semi_minor_confidence{0}; // Smaller eigenvalue of position covariance matrix.
    uint16_t position_semi_major_orientation{0}; // Direction of eigenvector of position covariance matrix.

    /* Heading confidence with precision 0.1 degrees from North. */
    uint16_t heading_c{0}; // Variance of heading measurements.

    /* Speed confidence with precision 0.01 m/s. */
    uint16_t speed_c{0}; // Variance of speed measurements.

    /* Acceleration confidence with precision 0.1 m/(s^2). */
    uint16_t acceleration_c{0}; // Variance of acceleration measurements.

    /* Yaw rate confidence with precision 0.1 degrees/s. */
    uint16_t yaw_rate_c{0}; // Variance of yaw rate measurements.

    /* Size confidence with precision 0.1m. */
    float length_c{0}; // Variance of RU length measurements.
    float width_c{0}; // Variance of RU width measurements.
    float height_c{0}; // Variance of RU height measurements.

    std::string signature{""}; // Used for signing the message content.
    std::string source_uuid{""};

    /* flag indicating availability of road user to receive waypoint */
    bool processing_waypoint{false};
    time_t waypoint_timestamp{0};

public:

    RoadUser(   std::string type,std::string context,std::string origin,std::string version,uint64_t timestamp,std::string uuid,std::string its_station_type,
                bool connected,int32_t latitude,int32_t longitude,std::string position_type,
                uint16_t heading,uint16_t speed,uint16_t acceleration,uint16_t yaw_rate,float length,float width,
                float height,std::string color,uint4 lane_position,uint8_t existence_probability,uint16_t position_semi_major_confidence,
                uint16_t position_semi_minor_confidence,uint16_t position_semi_major_orientation,uint16_t heading_c,
                uint16_t speed_c,int16_t acceleration_c,int16_t yaw_rate_c,float length_c,float width_c,float height_c,
                std::string signature, std::string source_uuid );
    RoadUser() = default;

    ~RoadUser() = default; // Destructor declaration.

    friend std::ostream& operator<< (ostream& os, RoadUser * roadUser); // Overload << to display a RoadUser.

    std::string getType();
    std::string getContext();
    std::string getOrigin();
    std::string getVersion();
    uint64_t getTimestamp();
    std::string getUuid();
    std::string getItsStationType();
    bool getConnected();
    int32_t getLatitude();
    int32_t getLongitude();
    uint16_t getHeading();
    uint16_t getSpeed();
    int16_t getAcceleration();
    int16_t getYawRate();
    float getLength();
    float getWidth();
    float getHeight();
    std::string getColor();
    uint4 getLanePosition();
    uint8_t getExistenceProbability();
    uint16_t getPositionSemiMajorConfidence();
    uint16_t getPositionSemiMinorConfidence();
    uint16_t getPositionSemiMajorOrientation();
    uint16_t getHeadingConfidence();
    uint16_t getSpeedConfidence();
    int16_t getAccelerationConfidence();
    int16_t getYawRateConfidence();
    float getLengthConfidence();
    float getWidthConfidence();
    float getHeightConfidence();
    std::string getSignature();
    std::string getPositionType();
    std::string getSouceUUID();
    bool getProcessingWaypoint();
    time_t getWaypointTimestamp();

    void setType(std::string);
    void setContext(std::string);
    void setOrigin(std::string);
    void setVersion(std::string);
    void setTimestamp(uint64_t);
    void setUuid(std::string);
    void setItsStationType(std::string);
    void setConnected(bool);
    void setLatitude(int32_t);
    void setLongitude(int32_t);
    void setHeading(uint16_t);
    void setSpeed(uint16_t);
    void setAcceleration(int16_t);
    void setYawRate(int16_t);
    void setLength(float);
    void setWidth(float);
    void setHeight(float);
    void setColor(std::string);
    void setLanePosition(uint4);
    void setExistenceProbability(uint8_t);
    void setPositionSemiMajorConfidence(uint16_t);
    void setPositionSemiMinorConfidence(uint16_t);
    void setPositionSemiMajorOrientation(uint16_t);
    void setHeadingConfidence(uint16_t);
    void setSpeedConfidence(uint16_t);
    void setAccelerationConfidence(int16_t);
    void setYawRateConfidence(int16_t);
    void setLengthConfidence(float);
    void setWidthConfidence(float);
    void setHeightConfidence(float);
    void setSignature(std::string);
    void setPositionType(std::string);
    void setSourceUUID(std::string);
    void setProcessingWaypoint(bool);
    void setWaypointTimeStamp(time_t);

};

#endif
