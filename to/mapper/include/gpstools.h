//
// Created by jab on 05/06/19.
//

#ifndef TO_GPSTOOLS_H
#define TO_GPSTOOLS_H

#include <string>

#define EARTH_RADIUS 6371000

struct Gps_Point {
    double latitude;
    double longitude;
};

struct Timebase_Telemetry_Waypoint {
    std::string id;
    Gps_Point coordinates;
    long timestamp;
    double heading;
    double speed;
    double accelleration;
    double yaw_rate;
};

#endif //TO_GPSTOOLS_H
