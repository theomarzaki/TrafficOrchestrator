// this file represents the secondary actions that the TO can send in order to create a sucessful merge

// main safety distances (presented in KPI) by manipulating other cars in the scenario.

// minimal changes applied to other cars as not to 'disturb' road equilibrium

#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>
#include <memory>
#include <vector>
#include <time.h>

using namespace std;
using namespace std::chrono;

const int SAFETY_DISTANCE = 30;

double distanceRoadUser(const std::shared_ptr<RoadUser> first,const std::shared_ptr<RoadUser> second){
  return distanceEarth(first->getLatitude(),first->getLongitude(),second->getLatitude(),second->getLongitude()) / 100;
}


std::vector<std::shared_ptr<ManeuverRecommendation>> calculateSafetyAction(std::shared_ptr<Database> database,vector<pair<std::shared_ptr<RoadUser>,
vector<std::shared_ptr<RoadUser>>>> neighbours){
  auto safetyManeuvers{std::vector<std::shared_ptr<ManeuverRecommendation>>()};

  auto timestamp = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

  for(const auto & neighbour : neighbours){
    auto scoped_scenario{getClosestFollowingandPreceedingCars(neighbour.first,neighbour.second, 0)}; // calculate preceeding and following cars for the same lane
    if(!scoped_scenario) continue;
    else{
      if(distanceRoadUser(neighbour.first,scoped_scenario->first) < SAFETY_DISTANCE && scoped_scenario->first->getConnected() && !(scoped_scenario->first->getProcessingWaypoint())){

        auto safetyManeuver{std::make_shared<ManeuverRecommendation>()};

        safetyManeuver->setTimestamp(timestamp);
        safetyManeuver->setUuidVehicle(scoped_scenario->first->getUuid());
        safetyManeuver->setUuidTo(scoped_scenario->first->getUuid());
        safetyManeuver->setTimestampAction(timestamp);
        safetyManeuver->setLongitudeAction(scoped_scenario->first->getLongitude());
        safetyManeuver->setLatitudeAction(scoped_scenario->first->getLatitude());
        safetyManeuver->setSpeedAction(ProcessedSpeedtoRoadUserSpeed(scoped_scenario->first->getSpeed()));
        safetyManeuver->setLanePositionAction(scoped_scenario->first->getLanePosition());
        safetyManeuver->setMessageID(std::string(safetyManeuver->getOrigin()) + "/" + std::string(safetyManeuver->getUuidManeuver()) + "/" +
                                      std::string(to_string(safetyManeuver->getTimestamp())));

        safetyManeuver->addWaypoint(RoadUser_action::accelerate(scoped_scenario->first));
        safetyManeuvers.emplace_back(safetyManeuver);

      }
      if(distanceRoadUser(neighbour.first,scoped_scenario->second) < SAFETY_DISTANCE && scoped_scenario->second->getConnected() && !(scoped_scenario->second->getProcessingWaypoint())){


        auto safetyManeuver{std::make_shared<ManeuverRecommendation>()};

        safetyManeuver->setTimestamp(timestamp);
        safetyManeuver->setUuidVehicle(scoped_scenario->second->getUuid());
        safetyManeuver->setUuidTo(scoped_scenario->second->getUuid());
        safetyManeuver->setTimestampAction(timestamp);
        safetyManeuver->setLongitudeAction(scoped_scenario->second->getLongitude());
        safetyManeuver->setLatitudeAction(scoped_scenario->second->getLatitude());
        safetyManeuver->setSpeedAction(ProcessedSpeedtoRoadUserSpeed(scoped_scenario->second->getSpeed()));
        safetyManeuver->setLanePositionAction(scoped_scenario->second->getLanePosition());
        safetyManeuver->setMessageID(std::string(safetyManeuver->getOrigin()) + "/" + std::string(safetyManeuver->getUuidManeuver()) + "/" +
                                      std::string(to_string(safetyManeuver->getTimestamp())));

        safetyManeuver->addWaypoint(RoadUser_action::deccelerate(scoped_scenario->second));
        safetyManeuvers.emplace_back(safetyManeuver);
      }
    }
  }

return safetyManeuvers;
}


std::vector<std::shared_ptr<ManeuverRecommendation>> stabiliseRoad(std::shared_ptr<Database> database) {
    auto recommendations{vector<std::shared_ptr<ManeuverRecommendation>>()};
    const auto road_users{database->findAll()};
    for (const auto &r : road_users) {
  			if (r->getLanePosition() != 0) {
            auto neighbours{mapNeighbours(database, 10000)};
            auto safety_action = calculateSafetyAction(database,neighbours);
            if(safety_action.size() != 0) {
              recommendations.insert(std::end(recommendations), std::begin(safety_action), std::end(safety_action));
            }
        }
    }
    return recommendations;
}
