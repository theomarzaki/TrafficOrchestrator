// This script provides the closest cars in terms of longitude and latitude

// Created by: KCL
#ifndef TO_NEAREST_NEIGHTBOUR_H
#define TO_NEAREST_NEIGHTBOUR_H

#include <iostream>
#include <vector>
#include <cmath>
#include <database.h>

#define EARTH_RADIUS_KM 6371.0

using std::vector;
using std::pair;

double rad2deg(double rad);

double deg2rad(double deg);

double distanceEarth(double lat1d, double lon1d, double lat2d, double lon2d);

auto mapNeighbours(const std::shared_ptr<Database>& database, double distanceRadius) -> vector<pair<std::shared_ptr<RoadUser>, vector<std::shared_ptr<RoadUser>>>>;

double calculateHeading(double lat1, double long1, double lat2, double long2);

#endif
