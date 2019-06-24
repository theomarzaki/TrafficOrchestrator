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
#include "gpsUtils.h"

using namespace GpsUtils;

auto a = 6378137;           // WGS-84 Earth semimajor axis (m)
auto b = 6356752.3142;      // WGS-84 Earth semiminor axis (m)
auto f = (a - b) / a;           // Ellipsoid Flatness
auto e_sq = f * (2 - f); // Square of Eccentricity
#define M_PIf64repl 3.141592653589793238462643383279502884 // Missing value from math.h 2.24 waiting to pass on 2.27

// Converts WGS-84 Geodetic point (lat, lon, h) to the
// Earth-Centered Earth-Fixed (ECEF) coordinates (x, y, z).
auto GpsUtils::geodeticToEcef(GpsCoordinates geodetic) -> ECEF {
    // Convert to radians in notation consistent with the paper:
    auto lambda = degreeToRadian(geodetic.latitude);
    auto phi = degreeToRadian(geodetic.longitude);
    auto s = std::sin(lambda);
    auto N = a / std::sqrt(1 - e_sq * s * s);

    ECEF out{};
    out.x = (geodetic.altitude + N) * std::cos(lambda) * std::cos(phi);
    out.y = (geodetic.altitude + N) * std::cos(lambda) * std::sin(phi);
    out.z = (geodetic.altitude + (1 - e_sq) * N) * std::sin(lambda);

    return out;
}

// Converts the Earth-Centered Earth-Fixed (ECEF) coordinates (x, y, z) to
// East-North-Up coordinates in a Local Tangent Plane that is centered at the
// (WGS-84) Geodetic point (lat0, lon0, h0).
auto GpsUtils::ecefToEnu(ECEF ecef, GpsCoordinates gpsOrigin) -> ENU {
    // Convert to radians in notation consistent with the paper:
    auto lambda = degreeToRadian(gpsOrigin.latitude);
    auto phi = degreeToRadian(gpsOrigin.longitude);
    auto s = std::sin(lambda);
    auto N = a / std::sqrt(1 - e_sq * s * s);

    double x0 = (gpsOrigin.altitude + N) * std::cos(lambda) * std::cos(phi);
    double y0 = (gpsOrigin.altitude + N) * std::cos(lambda) * std::sin(phi);
    double z0 = (gpsOrigin.altitude + (1 - e_sq) * N) * std::sin(lambda);

    double xd, yd, zd;
    xd = ecef.x - x0;
    yd = ecef.y - y0;
    zd = ecef.z - z0;
    ENU enu{};
    // This is the matrix multiplication
    enu.xEast = -std::sin(phi) * xd + std::cos(phi) * yd;
    enu.yNorth = -std::cos(phi) * std::sin(lambda) * xd - std::sin(lambda) * std::sin(phi) * yd + std::cos(lambda) * zd;
    enu.zUp = std::cos(lambda) * std::cos(phi) * xd + std::cos(lambda) * std::sin(phi) * yd + std::sin(lambda) * zd;
    return enu;
}

// Converts the geodetic WGS-84 coordinated (lat, lon, h) to
// East-North-Up coordinates in a Local Tangent Plane that is centered at the
// (WGS-84) Geodetic point (lat0, lon0, h0).
auto GpsUtils::geodeticToEnu(GpsCoordinates gpsToProject, GpsCoordinates gpsOrigin) -> ENU {
    ECEF ecef = geodeticToEcef(gpsToProject);
    return ecefToEnu(ecef, gpsOrigin);
}

auto GpsUtils::degreeToRadian(double angle) -> double {
    return M_PIf64repl * angle / 180.0;
}

double GpsUtils::toDegree(double value) {
    return value * 180.0 / M_PIf64repl;
}


