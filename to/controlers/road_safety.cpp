// this file represents the secondary actions that the TO can send in order to create a sucessful merge

// main safety distances (presented in KPI) by manipulating other cars in the scenario.

// minimal changes applied to other cars as not to 'disturb' road equilibrium

#include "include/road_safety.h"

#include <math.h>
#include <time.h>

#include <nearest_neighbour.h>
#include <create_trajectory.h>


const int SAFETY_DISTANCE = 10;

double distanceRoadUser(const std::shared_ptr<RoadUser>& first, const std::shared_ptr<RoadUser>& second) {
  return distanceEarth(first->getLatitude(),first->getLongitude(),second->getLatitude(),second->getLongitude()) / 100;
}


std::vector<std::shared_ptr<ManeuverRecommendation>> calculateSafetyAction( vector<pair<std::shared_ptr<RoadUser>, vector<std::shared_ptr<RoadUser>>>> neighbours) {
  auto safetyManeuver{std::vector<std::shared_ptr<ManeuverRecommendation>>()};

  for(const auto & neighbour : neighbours){
      auto scoped_scenario{getClosestFollowingandPreceedingCars(neighbour.first,neighbour.second, 0)}; // calculate preceeding and following cars for the same lane
      if(!scoped_scenario) continue;
      else {
          if(distanceRoadUser(neighbour.first,scoped_scenario->first) < SAFETY_DISTANCE){ //preceeding car
              std::cout << neighbour.first->getUuid() << distanceRoadUser(neighbour.first,scoped_scenario->first) << std::endl;
          }
          if(distanceRoadUser(neighbour.first,scoped_scenario->second) < SAFETY_DISTANCE){ //following car
              std::cout << neighbour.first->getUuid() << distanceRoadUser(neighbour.first,scoped_scenario->second) << std::endl;
          }
      }
  }

  return safetyManeuver;
}


std::vector<std::shared_ptr<ManeuverRecommendation>> stabiliseRoad(std::shared_ptr<Database> database) {
    auto recommendations{vector<std::shared_ptr<ManeuverRecommendation>>()};
    const auto road_users{database->findAll()};
    for (const auto &r : road_users) {
        if(difftime(time(nullptr),r->getWaypointTimestamp()) < 0){
            r->setProcessingWaypoint(false);
            database->upsert(r);
        }
        if (r->getConnected() && !(r->getProcessingWaypoint()) && r->getLanePosition() != 0) {
            auto neighbours{mapNeighbours(database)};
            auto safety_action = calculateSafetyAction(neighbours);
            if(!safety_action.empty()) {
                recommendations.insert(std::end(recommendations), std::begin(safety_action), std::end(safety_action));
            }
        }
    }
    return recommendations;
}
