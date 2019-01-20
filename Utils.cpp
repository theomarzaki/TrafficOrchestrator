#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <nearest_neighbour.cpp>
#include <torch/torch.h>
#include <torch/script.h>

using namespace std;

using namespace rapidjson;

using namespace std::chrono;
using std::cout;


vector<float> RoadUsertoModelInput(vector<pair<RoadUser*,vector<RoadUser*>>> neighbours){
  std::vector<float> mergingCar;
  std::vector<float> preceeding_car;
  std::vector<float> following_car;
  // concatenate the lists into one list for the lstm
  for(pair<RoadUser*,vector<RoadUser*>> v : neighbours){

    v.first;

    if(v.second.size() == 0){
      continue;
    }else if(v.second.size() == 1){
      continue;
    }else if(v.second.size() == 2){
      continue;
    }else{
      continue;
    }

    mergingCar.insert( mergingCar.end(), preceeding_car.begin(), preceeding_car.end() );
    mergingCar.insert( mergingCar.end(), following_car.begin(), following_car.end() );

  }
  return mergingCar;
}

ManeuverRecommendation* calculatedTrajectories(RoadUser * mergingVehicle,at::Tensor calculatedLSTM ){
  ManeuverRecommendation* mergingManeuver = new ManeuverRecommendation();

  auto timeCalculator = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  mergingManeuver->setTimestamp(timeCalculator.count());
  mergingManeuver->setUuidVehicle(mergingVehicle->getUuid());
  mergingManeuver->setUuidTo(mergingVehicle->getUuid());
  mergingManeuver->setTimestampAction(timeCalculator.count());
  mergingManeuver->setLongitudeAction(mergingVehicle->getLongitude());
  mergingManeuver->setLatitudeAction(mergingVehicle->getLatitude());
  mergingManeuver->setSpeedAction(mergingVehicle->getSpeed());
  mergingManeuver->setLanePositionAction(mergingVehicle->getLanePosition());

  Waypoint * waypoint = new Waypoint();
  waypoint->setTimestamp(timeCalculator.count() + (90/mergingVehicle->getSpeed())*1000);
  waypoint->setLatitude(969);
  waypoint->setLongitude(696);
  waypoint->setSpeed(mergingVehicle->getSpeed());
  waypoint->setLanePosition(mergingVehicle->getLanePosition()+1);
  mergingManeuver->addWaypoint(waypoint);
  // obtain list, break it down into the respective components and assign the recomendation feedback
  return mergingManeuver;
}

vector<ManeuverRecommendation*> ManeuverParser(Database * database, double distanceRadius,std::shared_ptr<torch::jit::script::Module> lstm_model){
  vector<ManeuverRecommendation*> recommendations;
  for(RoadUser * r : *database->getDatabase()) {
		if(r->getConnected() == true && r->getLanePosition() == 0) {
			printf("CAR IN LANE 0.\n");
			auto neighbours = mapNeighbours(database,distanceRadius);
      auto input_values = RoadUsertoModelInput(neighbours);
      std::vector<torch::jit::IValue> lstm_inputs;
      auto makeshift = torch::rand({1, 2, 19});
      // makeshift[1][1] = torch::tensor(input_values);
      lstm_inputs.push_back(makeshift);
      recommendations.push_back(calculatedTrajectories(r,lstm_model->forward(lstm_inputs).toTensor()));
			printf("SEEING IF THERE IS FOLLOWING / PRECEEDING CAR.\n");
      cout << "number of neighbours :" << neighbours.size() << endl;
		}
	}
  return recommendations;
}
