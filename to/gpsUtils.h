//
// Created by Elias Salam<elias.salam@orange.com> on 11/6/19.
//
// Some helpers for converting GPS readings from the WGS84 geodetic system to a local North-East-Up cartesian axis.

// The implementation here is according to the paper:
// "Conversion of Geodetic coordinates to the Local Tangent Plane" Version 2.01.
// "The basic reference for this paper is J.Farrell & M.Barth 'The Global Positioning System & Inertial Navigation'"
// Also helpful is Wikipedia: http://en.wikipedia.org/wiki/Geodetic_datum
// Taken from https://gist.github.com/govert/1b373696c9a27ff4c72a


#include <math.h>

namespace GpsUtils {

    struct ECEF {
        double x;
        double y;
        double z;
    };

    struct ENU {
        double xEast;
        double yNorth;
        double zUp;
    };

    struct GpsCoordinates {
        double altitude{0.0};
        double latitude{0.0};
        double longitude{0.0};
    };

    auto degreeToRadian(double angle) -> double;

    // Converts WGS-84 Geodetic point (lat, lon, h) to the
    // Earth-Centered Earth-Fixed (ECEF) coordinates (x, y, z).
    auto geodeticToEcef(GpsCoordinates geodetic) -> ECEF;

    // Converts the Earth-Centered Earth-Fixed (ECEF) coordinates (x, y, z) to
    // East-North-Up coordinates in a Local Tangent Plane that is centered at the
    // (WGS-84) Geodetic point (lat0, lon0, h0).
    auto ecefToEnu(ECEF ecef, GpsCoordinates gpsOrigin) -> ENU;

    // Converts the geodetic WGS-84 coordinated (lat, lon, h) to
    // East-North-Up coordinates in a Local Tangent Plane that is centered at the
    // (WGS-84) Geodetic point (lat0, lon0, h0).
    auto geodeticToEnu(GpsCoordinates gpsToProject, GpsCoordinates gpsOrigin) -> ENU;

    auto degreeToRadian(double angle) -> double;

    auto toDegree(double value) -> double;

}
