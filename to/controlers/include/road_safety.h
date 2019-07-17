//
// Created by jab on 16/07/19.
//

#ifndef TO_ROAD_SAFETY_H
#define TO_ROAD_SAFETY_H


#include <memory>
#include <vector>
#include <road_user.h>
#include <maneuver_recommendation.h>
#include <database.h>

double distanceRoadUser(const std::shared_ptr<RoadUser>& first, const std::shared_ptr<RoadUser>& second);

std::vector<std::shared_ptr<ManeuverRecommendation>> calculateSafetyAction( std::vector<std::pair<std::shared_ptr<RoadUser>, std::vector<std::shared_ptr<RoadUser>>>> neighbours);

std::vector<std::shared_ptr<ManeuverRecommendation>> stabiliseRoad(std::shared_ptr<Database> database);

#endif //TO_ROAD_SAFETY_H
