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

struct Point_2D {
    double x;
    double y;
};

struct Timebase_Telemetry_Waypoint {
    Gps_Point coordinates;
    std::string uuid;
    bool connected;
    int laneId;
    int64_t timestamp;
    double heading;
    double speed;
    double acceleration;
    double yaw_rate;
    double max_speed;
    double mass;
};

struct Gps_View {
    double latitude;
    double longitude;
    double heading;
};

#endif //TO_GPSTOOLS_H
