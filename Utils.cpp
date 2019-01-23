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

double distanceCalculate(double x1, double y1, double x2, double y2)
{
	double x = x1 - x2; //calculating number to square in next step
	double y = y1 - y2;
	double dist;

	dist = x*x + y*y;       //calculating Euclidean distance
	dist = sqrt(dist);

	return dist;
}

pair<RoadUser*,RoadUser*> getClosestFollowingandPreceedingCars(RoadUser * merging_car,std::vector<RoadUser*> close_by){
  RoadUser * closest_following = new RoadUser();
  closest_following->setLongitude(merging_car->getLongitude()-10000); //check TODO
  closest_following->setLatitude(merging_car->getLatitude()-10000); // check
  closest_following->setSpeed(merging_car->getSpeed());
  closest_following->setWidth(merging_car->getWidth());
  closest_following->setLength(merging_car->getLength());
  closest_following->setAcceleration(merging_car->getAcceleration());
  closest_following->setLanePosition(merging_car->getLanePosition()+1);
  RoadUser * closest_preceeding = new RoadUser();
  closest_preceeding->setLongitude(merging_car->getLongitude()+10000); // check
  closest_preceeding->setLatitude(merging_car->getLatitude()+10000); //check
  closest_preceeding->setSpeed(merging_car->getSpeed());
  closest_preceeding->setWidth(merging_car->getWidth());
  closest_preceeding->setLength(merging_car->getLength());
  closest_preceeding->setAcceleration(merging_car->getAcceleration());
  closest_preceeding->setLanePosition(merging_car->getLanePosition()+1);

  int minFollowing = 9999;
  int minPreceeding = 9999;

  for(RoadUser * close_car : close_by){
    if(close_car->getLatitude() < merging_car->getLatitude() && close_car->getLongitude() < merging_car->getLongitude()){ //closest following
      if(distanceCalculate(close_car->getLatitude(),close_car->getLongitude(),merging_car->getLatitude(),merging_car->getLongitude()) < minFollowing){
          closest_following = close_car;
      }
    }
    if (close_car->getLatitude() > merging_car->getLatitude() && close_car->getLongitude() > merging_car->getLongitude()){ //closest preceeding
      if(distanceCalculate(close_car->getLatitude(),close_car->getLongitude(),merging_car->getLatitude(),merging_car->getLongitude()) < minPreceeding){
        closest_preceeding = close_car;
      }
    }
  }
return pair<RoadUser*,RoadUser*>(closest_preceeding,closest_following);



}

vector<float> RoadUsertoModelInput(RoadUser * merging_car,vector<pair<RoadUser*,vector<RoadUser*>>> neighbours){
  std::vector<float> mergingCar;

  printf("SEEING IF THERE IS FOLLOWING / PRECEEDING CAR.\n");
  cout << "number of neighbours :" << neighbours.size() << endl;
  std::vector<RoadUser*> v;
  auto x = getClosestFollowingandPreceedingCars(merging_car,v);
  // concatenate the lists into one list for the lstm
  for(pair<RoadUser*,vector<RoadUser*>> v : neighbours){
    if ( v.first->getUuid() == merging_car->getUuid() ){
      cout << "WOHOOOOOO" << endl;
      x = getClosestFollowingandPreceedingCars(merging_car,v.second);
      cout << x.first->getLongitude() << x.first->getLatitude() << x.second->getLongitude() << x.second->getLatitude() << endl;
    } else cout << "no nearby cars" << endl;
  }

    mergingCar.push_back(merging_car->getLatitude());
    mergingCar.push_back(merging_car->getLongitude());
    mergingCar.push_back(merging_car->getLength());
    mergingCar.push_back(merging_car->getWidth());
    mergingCar.push_back(merging_car->getSpeed());
    mergingCar.push_back(merging_car->getAcceleration());
    mergingCar.push_back(merging_car->getLatitude()); // spacing
    mergingCar.push_back(x.first->getLatitude());
    mergingCar.push_back(x.first->getLongitude());
    mergingCar.push_back(x.first->getLength());
    mergingCar.push_back(x.first->getWidth());
    mergingCar.push_back(x.first->getSpeed());
    mergingCar.push_back(x.first->getAcceleration());
    mergingCar.push_back(x.second->getLatitude());
    mergingCar.push_back(x.second->getLongitude());
    mergingCar.push_back(x.second->getWidth());
    mergingCar.push_back(x.second->getSpeed());
    mergingCar.push_back(x.second->getAcceleration());
    mergingCar.push_back(x.second->getLatitude()); // spacing

  return mergingCar;
}

ManeuverRecommendation* calculatedTrajectories(RoadUser * mergingVehicle,at::Tensor models_input,std::shared_ptr<torch::jit::script::Module> lstm_model,std::shared_ptr<torch::jit::script::Module> rl_model){
  ManeuverRecommendation* mergingManeuver = new ManeuverRecommendation();
	std::vector<torch::jit::IValue> lstm_inputs;
	std::vector<torch::jit::IValue> rl_inputs;

	auto timeCalculator = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  mergingManeuver->setTimestamp(timeCalculator.count());
  mergingManeuver->setUuidVehicle(mergingVehicle->getUuid());
  mergingManeuver->setUuidTo(mergingVehicle->getUuid());
  mergingManeuver->setTimestampAction(timeCalculator.count());
  mergingManeuver->setLongitudeAction(mergingVehicle->getLongitude());
  mergingManeuver->setLatitudeAction(mergingVehicle->getLatitude());
  mergingManeuver->setSpeedAction(mergingVehicle->getSpeed());
  mergingManeuver->setLanePositionAction(mergingVehicle->getLanePosition());

	lstm_inputs.push_back(models_input);
	at::Tensor calculatedLSTM = lstm_model->forward(lstm_inputs).toTensor();
	rl_inputs.push_back(calculatedLSTM);
	at::Tensor calculatedRL = rl_model->forward(rl_inputs).toTensor();

	// repeat for 1-10 times depending on number of waypoints you want
	// parse action into waypoint
	// add more waypoints here
  Waypoint * waypoint = new Waypoint();
  waypoint->setTimestamp(timeCalculator.count() + (distanceCalculate(mergingVehicle->getLatitude(),mergingVehicle->getLongitude(),calculatedLSTM[0][0].item<float>(),calculatedLSTM[0][1].item<float>())/mergingVehicle->getSpeed())*1000); //distance to mergeing point
  waypoint->setLatitude(calculatedLSTM[0][0].item<float>());
  waypoint->setLongitude(calculatedLSTM[0][1].item<float>());
  waypoint->setSpeed(calculatedLSTM[0][4].item<float>());
  waypoint->setLanePosition(mergingVehicle->getLanePosition()+1);
  mergingManeuver->addWaypoint(waypoint);

  return mergingManeuver;
}

vector<ManeuverRecommendation*> ManeuverParser(Database * database, double distanceRadius,std::shared_ptr<torch::jit::script::Module> lstm_model,std::shared_ptr<torch::jit::script::Module> rl_model){
  vector<ManeuverRecommendation*> recommendations;
  for(RoadUser * r : *database->getDatabase()) {
		if(r->getConnected() == true && r->getLanePosition() == 0) {
			printf("CAR IN LANE 0.\n");
			auto neighbours = mapNeighbours(database,distanceRadius);
      auto input_values = RoadUsertoModelInput(r,neighbours);
      auto models_input = torch::tensor(input_values).unsqueeze(0).unsqueeze(0);;
      recommendations.push_back(calculatedTrajectories(r,models_input,lstm_model,rl_model));
		}
	}
  return recommendations;
}
