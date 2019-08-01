// This script provides the closest cars in terms of longitude and latitude

// Created by: KCL
#include "nearest_neighbour.h"

double rad2deg(double rad) {
  return (rad * 180 / M_PI);
}

double deg2rad(double deg) {
  return (deg * M_PI / 180);
}

double distanceEarth(double lat1d, double lon1d, double lat2d, double lon2d) {
  double lat1r, lon1r, lat2r, lon2r, u, v;
  lat1r = deg2rad(lat1d);
  lon1r = deg2rad(lon1d);
  lat2r = deg2rad(lat2d);
  lon2r = deg2rad(lon2d);
  u = sin((lat2r - lat1r)/2);
  v = sin((lon2r - lon1r)/2);
  return 2.0 * EARTH_RADIUS_KM * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
}

auto mapNeighbours(const std::shared_ptr<Database>& database) -> vector<pair<std::shared_ptr<RoadUser>, vector<std::shared_ptr<RoadUser>>>> {
	vector<pair<std::shared_ptr<RoadUser>, vector<std::shared_ptr<RoadUser>>>> neighbours;
	vector<std::shared_ptr<RoadUser>> closeBy;
	if (database->getSize() != 1) {
		const auto roadUsers{database->findAll()};
		for (unsigned long i = 0; i < roadUsers.size(); ++i) {
			for (unsigned long j = i; ++j != roadUsers.size();) {
				closeBy.push_back(roadUsers.at(j));
			}
			auto pair{make_pair(roadUsers.at(i), closeBy)};
			closeBy.clear();
			neighbours.push_back(pair);
		}
	}
	return neighbours;
}

double calculateHeading(double lat1, double long1, double lat2, double long2) {
	double a = lat1 * M_PI / 180;
    double b = long1 * M_PI / 180;
    double c = lat2 * M_PI / 180;
    double d = long2 * M_PI / 180;

    if(cos(c) * sin(d - b) == 0) {
    	if(c > a) {
    		return 0;
    	}
    	else {
    		return 180;
    	}
    }

    else {
    	double angle = atan2(cos(c) * sin(d-b), sin(c) * cos(a) - sin(a) * cos(c) * cos(d-b));
    	return std::fmod(angle*180/M_PI+360,360);
    }
}
