// this file represents the secondary actions that the TO can send in order to create a sucessful merge

// main safety distances (presented in KPI) by manipulating other cars in the scenario.

// minimal changes applied to other cars as not to 'disturb' road equilibrium

#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>
#include <memory>
#include <vector>
#include <time.h>

#include <nearest_neighbour.h>
#include <maneuver_recommendation.h>
#include <waypoint.h>
#include <mapper.h>
#include <database.h>
#include <create_trajectory.h>

using namespace std;
using namespace std::chrono;

const int SAFETY_DISTANCE = 10;

double distanceRoadUser(const std::shared_ptr<RoadUser> first,const std::shared_ptr<RoadUser> second){
  return distanceEarth(first->getLatitude(),first->getLongitude(),second->getLatitude(),second->getLongitude()) / 100;
}


std::vector<std::shared_ptr<ManeuverRecommendation>> calculateSafetyAction(std::shared_ptr<Database> database,vector<pair<std::shared_ptr<RoadUser>,
vector<std::shared_ptr<RoadUser>>>> neighbours){
  auto safetyManeuver{std::vector<std::shared_ptr<ManeuverRecommendation>>()};

  for(const auto & neighbour : neighbours){
    auto scoped_scenario{getClosestFollowingandPreceedingCars(neighbour.first,neighbour.second, 0)}; // calculate preceeding and following cars for the same lane
    if(!scoped_scenario) continue;
    else{
      if(distanceRoadUser(neighbour.first,scoped_scenario->first) < SAFETY_DISTANCE){ //preceeding car
        cout << neighbour.first->getUuid() << distanceRoadUser(neighbour.first,scoped_scenario->first) << endl;
      }
      if(distanceRoadUser(neighbour.first,scoped_scenario->second) < SAFETY_DISTANCE){ //following car
        cout << neighbour.first->getUuid() << distanceRoadUser(neighbour.first,scoped_scenario->second) << endl;
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
	            auto neighbours{mapNeighbours(database, 10000)};
              auto safety_action = calculateSafetyAction(database,neighbours);
              if(safety_action.size() != 0) {
                recommendations.insert(std::end(recommendations), std::begin(safety_action), std::end(safety_action));
              }
	        }
    }
    return recommendations;
}
