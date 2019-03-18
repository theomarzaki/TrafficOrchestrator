// This file uses both of the models(LSTM,RL) to compute way points for a 3 - window

// returns a maneuver recommendation to the TO

// Calculated actions to waypoints and vice versa

// change to n step window

// Created by: Omar Nassef (KCL)


#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <to/nearest_neighbour.cpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>
#include <memory>

using namespace std;
using namespace rapidjson;

int TIME_VARIANT = 0.035;

using namespace std::chrono;
using std::cout;

int RoadUserSpeedtoProcessedSpeed(int speed){
	return speed * 100;
}

int ProcessedSpeedtoRoadUserSpeed(int speed){
	return speed / 100;
}

float RoadUserGPStoProcessedGPS(float point){
	return point / pow(10,6);
}
float ProcessedGPStoRoadUserGPS(float point){
	return point * pow(10,6);
}

bool inRange(int low, int high, int x){
    return ((x-high)*(x-low) <= 0);
}

// bool isCarTerminal(at::Tensor state){
//   float y_diff = state[0][14].item<float>() - state[0][8].item<float>();
//   float x_diff = state[0][13].item<float>() - state[0][7].item<float>();
//
//   try{
//     float slope = round(y_diff) / round(x_diff);
//     if(isinf(slope) || isnan(slope)) slope = 0;
//     float plus_c = state[0][8].item<float>() - (slope * state[0][7].item<float>());
//     if(!isinf(state[0][0].item<float>()) && !isinf(state[0][1].item<float>())){
//         if(inRange(round(slope * int(state[0][0].item<float>()) + plus_c) - 1, round(slope * int(state[0][0].item<float>()) + plus_c) + 1,round(int(state[0][1].item<float>())))){
//             if(int(state[0][7].item<float>()) > int(state[0][0].item<float>()) && int(state[0][0].item<float>()) > int(state[0][13].item<float>()) && int(state[0][8].item<float>()) < int(state[0][1].item<float>()) && int(state[0][1].item<float>()) < int(state[0][14].item<float>())){
//                 return true;
// 						}
// 				}
// 		}
// 	}
//   catch(...){
//     float plus_c = int(state[0][8].item<float>());
//     if ((round(state[0][1].item<float>()) + 1 == round(plus_c) or round(state[0][1].item<float>()) - 1 == round(plus_c))){
//          if(int(state[0][7].item<float>()) > int(state[0][0].item<float>()) && int(state[0][0].item<float>()) > int(state[0][13].item<float>()) && int(state[0][8].item<float>()) < int(state[0][1].item<float>()) && int(state[0][1].item<float>()) < int(state[0][14].item<float>())){
//              return true;
// 					 }
// 		}
// 	}
//   return false;
// }
// TODO

// 1. spacing for cars (remove for shorter training time)


pair<RoadUser*,RoadUser*> getClosestFollowingandPreceedingCars(RoadUser * merging_car,std::vector<RoadUser*> close_by){
  RoadUser * closest_following = new RoadUser();
  closest_following->setLongitude(RoadUserGPStoProcessedGPS(merging_car->getLongitude()-10000)); //check TODO
  closest_following->setLatitude(RoadUserGPStoProcessedGPS(merging_car->getLatitude()-10000)); // check
  closest_following->setSpeed(merging_car->getSpeed());
  closest_following->setWidth(merging_car->getWidth());
  closest_following->setLength(merging_car->getLength());
  closest_following->setAcceleration(merging_car->getAcceleration());
  closest_following->setLanePosition(merging_car->getLanePosition()+1);
  RoadUser * closest_preceeding = new RoadUser();
  closest_preceeding->setLongitude(RoadUserGPStoProcessedGPS(merging_car->getLongitude()+10000)); // check
  closest_preceeding->setLatitude(RoadUserGPStoProcessedGPS(merging_car->getLatitude()+10000)); //check
  closest_preceeding->setSpeed(merging_car->getSpeed());
  closest_preceeding->setWidth(merging_car->getWidth());
  closest_preceeding->setLength(merging_car->getLength());
  closest_preceeding->setAcceleration(merging_car->getAcceleration());
  closest_preceeding->setLanePosition(merging_car->getLanePosition()+1);

  int minFollowing = 9999;
  int minPreceeding = 9999;

  for(RoadUser * close_car : close_by){
    if(close_car->getLatitude() < merging_car->getLatitude() && close_car->getLongitude() < merging_car->getLongitude()){ //closest following
      if(distanceEarth(RoadUserGPStoProcessedGPS(close_car->getLatitude()),RoadUserGPStoProcessedGPS(close_car->getLongitude()),RoadUserGPStoProcessedGPS(merging_car->getLatitude()),RoadUserGPStoProcessedGPS(merging_car->getLongitude())) < minFollowing){
          closest_following = close_car;
      }
    }
    if (close_car->getLatitude() > merging_car->getLatitude() && close_car->getLongitude() > merging_car->getLongitude()){ //closest preceeding
      if(distanceEarth(RoadUserGPStoProcessedGPS(close_car->getLatitude()),RoadUserGPStoProcessedGPS(close_car->getLongitude()),RoadUserGPStoProcessedGPS(merging_car->getLatitude()),RoadUserGPStoProcessedGPS(merging_car->getLongitude())) < minPreceeding){
        closest_preceeding = close_car;
      }
    }
  }
return pair<RoadUser*,RoadUser*>(closest_preceeding,closest_following);
}

// For RL only Algorithm
at::Tensor GetStateFromActions(at::Tensor action_Tensor,at::Tensor stateTensor){
	int accelerate_tensor = 0;
	int deccelerate_tensor = 1;
	int left_tensor = 2;
	int right_tensor = 3;
	int doNothing_tensor = 4;

	auto state = stateTensor;

	auto actionTensor = torch::argmax(action_Tensor);
	if(accelerate_tensor == actionTensor.item<int>()){
			auto final_velocity = state[0][4].item<float>() + TIME_VARIANT * (state[0][4].item<float>() + state[0][5].item<float>() * TIME_VARIANT);
			auto final_acceleration = (pow(final_velocity,2) - pow(state[0][4].item<float>(),2)) / 2 * (0.5 * (state[0][4].item<float>() + final_velocity) * TIME_VARIANT);
			auto displacement = final_velocity * TIME_VARIANT + 0.5 * (final_acceleration * TIME_VARIANT * TIME_VARIANT);
			auto angle = int(state[0][19].item<float>());
			auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
			auto new_y = state[0][1].item<float>() + displacement * sin((angle * M_PI)/ 180);
			stateTensor[0][0] = new_x;
			stateTensor[0][1] = new_y;
			stateTensor[0][4] = final_velocity;
			stateTensor[0][5] = final_velocity;
			stateTensor[0][19] = angle;
		return stateTensor;
	} else if(deccelerate_tensor == actionTensor.item<int>()){
		auto final_velocity = state[0][4].item<float>() - TIME_VARIANT * (state[0][4].item<float>() + state[0][5].item<float>() * TIME_VARIANT);
		auto final_acceleration = (pow(final_velocity,2) - pow(state[0][4].item<float>(),2)) / 2 * (0.5 * (state[0][4].item<float>() + final_velocity) * TIME_VARIANT);
		auto displacement = final_velocity * TIME_VARIANT + 0.5 * (final_acceleration * TIME_VARIANT * TIME_VARIANT);
		auto angle = int(state[0][19].item<float>());
		auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
		auto new_y = state[0][1].item<float>() + displacement * sin((angle * M_PI)/ 180);
		stateTensor[0][0] = new_x;
		stateTensor[0][1] = new_y;
		stateTensor[0][4] = final_velocity;
		stateTensor[0][5] = final_velocity;
		stateTensor[0][19] = angle;
	return stateTensor;
	} else if(left_tensor == actionTensor.item<int>()){
			float displacement = state[0][4].item<float>() * TIME_VARIANT + 0.5 * (state[0][5].item<float>() * TIME_VARIANT * TIME_VARIANT);
			auto angle = int(state[0][19].item<float>());
			angle = (angle + 1) % 360;
			auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
			auto new_y = state[0][1].item<float>()  + displacement * sin((angle * M_PI)/ 180);
			stateTensor[0][0] = new_x;
			stateTensor[0][1] = new_y;
			stateTensor[0][19] = angle;
		return stateTensor;
	} else if(right_tensor == actionTensor.item<int>()){
			auto displacement = state[0][4].item<float>() * TIME_VARIANT + 0.5 * (state[0][5].item<float>() * TIME_VARIANT * TIME_VARIANT);
			auto angle = int(state[0][19].item<float>());
			angle = (angle - 1) % 360;
			auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
			auto new_y = state[0][1].item<float>()  + displacement * sin((angle * M_PI)/ 180);
			stateTensor[0][0] = new_x;
			stateTensor[0][1] = new_y;
			stateTensor[0][19] = angle;
		return stateTensor;
	} else if(doNothing_tensor == actionTensor.item<int>()){
			auto displacement = state[0][4].item<float>() * TIME_VARIANT + 0.5 * (state[0][5].item<float>() * TIME_VARIANT * TIME_VARIANT);
	   	auto angle = int(state[0][19].item<float>());
	    auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
	    auto new_y = state[0][1].item<float>() + displacement * sin((angle * M_PI)/ 180);
	    stateTensor[0][0] = new_x;
	    stateTensor[0][1] = new_y;
			stateTensor[0][19] = angle;
    return stateTensor;
	} else cout << "ERROR: incomputing incorrect action tensor";

	return stateTensor;
}




vector<float> RoadUsertoModelInput(RoadUser * merging_car,vector<pair<RoadUser*,vector<RoadUser*>>> neighbours){
  std::vector<float> mergingCar;

  std::vector<RoadUser*> v;
  auto x = getClosestFollowingandPreceedingCars(merging_car,v);
  for(pair<RoadUser*,vector<RoadUser*>> v : neighbours){
    if ( v.first->getUuid() == merging_car->getUuid() ){
      x = getClosestFollowingandPreceedingCars(merging_car,v.second);
  	}
	}

    mergingCar.push_back(RoadUserGPStoProcessedGPS(merging_car->getLatitude()));
    mergingCar.push_back(RoadUserGPStoProcessedGPS(merging_car->getLongitude()));
    mergingCar.push_back(merging_car->getLength());
    mergingCar.push_back(merging_car->getWidth());
    mergingCar.push_back(RoadUserSpeedtoProcessedSpeed(merging_car->getSpeed()));
    mergingCar.push_back(merging_car->getAcceleration());
    mergingCar.push_back(distanceEarth(RoadUserGPStoProcessedGPS(merging_car->getLongitude()),RoadUserGPStoProcessedGPS(merging_car->getLatitude()),RoadUserGPStoProcessedGPS(x.first->getLongitude()),RoadUserGPStoProcessedGPS(x.first->getLatitude()))); // spacing
    mergingCar.push_back(RoadUserGPStoProcessedGPS(x.first->getLatitude()));
    mergingCar.push_back(RoadUserGPStoProcessedGPS(x.first->getLongitude()));
    mergingCar.push_back(x.first->getLength());
    mergingCar.push_back(x.first->getWidth());
    mergingCar.push_back(RoadUserSpeedtoProcessedSpeed(x.first->getSpeed()));
    mergingCar.push_back(x.first->getAcceleration());
    mergingCar.push_back(RoadUserGPStoProcessedGPS(x.second->getLatitude()));
    mergingCar.push_back(RoadUserGPStoProcessedGPS(x.second->getLongitude()));
    mergingCar.push_back(x.second->getWidth());
    mergingCar.push_back(RoadUserSpeedtoProcessedSpeed(x.second->getSpeed()));
    mergingCar.push_back(x.second->getAcceleration());
    mergingCar.push_back(distanceEarth(RoadUserGPStoProcessedGPS(merging_car->getLongitude()),RoadUserGPStoProcessedGPS(merging_car->getLatitude()),RoadUserGPStoProcessedGPS(x.second->getLongitude()),RoadUserGPStoProcessedGPS(x.second->getLatitude()))); // spacing
		mergingCar.push_back(merging_car->getHeading());

  return mergingCar;
}

ManeuverRecommendation* calculatedTrajectories(RoadUser * mergingVehicle,at::Tensor models_input,std::shared_ptr<torch::jit::script::Module> lstm_model,std::shared_ptr<torch::jit::script::Module> rl_model){
  ManeuverRecommendation* mergingManeuver = new ManeuverRecommendation();
	std::vector<torch::jit::IValue> rl_inputs;

	auto timeCalculator = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  mergingManeuver->setTimestamp(timeCalculator.count());
  mergingManeuver->setUuidVehicle(mergingVehicle->getUuid());
  mergingManeuver->setUuidTo(mergingVehicle->getUuid());
  mergingManeuver->setTimestampAction(timeCalculator.count());
  mergingManeuver->setLongitudeAction(mergingVehicle->getLongitude());
  mergingManeuver->setLatitudeAction(mergingVehicle->getLatitude());
  mergingManeuver->setSpeedAction(ProcessedSpeedtoRoadUserSpeed(mergingVehicle->getSpeed()));
  mergingManeuver->setLanePositionAction(mergingVehicle->getLanePosition());
	mergingManeuver->setMessageID(std::string(mergingManeuver->getOrigin()) + "/" + std::string(mergingManeuver->getUuidManeuver()) + "/" + std::string(to_string(mergingManeuver->getTimestamp())));

	rl_inputs.push_back(models_input);
	at::Tensor calculatedRL = rl_model->forward(rl_inputs).toTensor();
	auto calculated_n_1_states = GetStateFromActions(calculatedRL,models_input);

	Waypoint * waypoint = new Waypoint();
  waypoint->setTimestamp(timeCalculator.count() + (distanceEarth(mergingVehicle->getLatitude(),mergingVehicle->getLongitude(),calculated_n_1_states[0][0].item<float>(),calculated_n_1_states[0][1].item<float>())/mergingVehicle->getSpeed())*1000); //distance to mergeing point
  waypoint->setLatitude(ProcessedGPStoRoadUserGPS(calculated_n_1_states[0][0].item<float>()));
  waypoint->setLongitude(ProcessedGPStoRoadUserGPS(calculated_n_1_states[0][1].item<float>()));
  waypoint->setSpeed(ProcessedSpeedtoRoadUserSpeed(calculated_n_1_states[0][4].item<float>()));
  waypoint->setLanePosition(mergingVehicle->getLanePosition());
  mergingManeuver->addWaypoint(waypoint);

	// at::Tensor previous_state = calculated_n_1_states;
	// for(int counter = 0;counter < 4; counter++){ //number of waypoints
	// 	auto timeCalculator = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	// 	std::vector<torch::jit::IValue> rl_n_inputs;
	//
	// 	rl_n_inputs.push_back(previous_state);
	// 	auto calculated_next_state = rl_model->forward(rl_n_inputs).toTensor();
	// 	auto calculated_waypoint = GetStateFromActions(calculated_next_state,previous_state);
	// 	previous_state = calculated_waypoint;
	//
	// 	Waypoint * n_waypoint = new Waypoint();
	//   n_waypoint->setTimestamp(timeCalculator.count() + (distanceEarth(mergingVehicle->getLatitude(),mergingVehicle->getLongitude(),calculated_waypoint[0][0].item<float>(),calculated_waypoint[0][1].item<float>())/mergingVehicle->getSpeed())*1000); //distance to mergeing point
	//   n_waypoint->setLatitude(ProcessedGPStoRoadUserGPS(calculated_waypoint[0][0].item<float>()));
	//   n_waypoint->setLongitude(ProcessedGPStoRoadUserGPS(calculated_waypoint[0][1].item<float>()));
	//   n_waypoint->setSpeed(ProcessedSpeedtoRoadUserSpeed(calculated_waypoint[0][4].item<float>()));
	//   n_waypoint->setLanePosition(mergingVehicle->getLanePosition());
	//   mergingManeuver->addWaypoint(n_waypoint);
	//
	// }
  return mergingManeuver;
}

vector<ManeuverRecommendation *>
ManeuverParser(Database *database, double distanceRadius, std::shared_ptr<torch::jit::script::Module> lstm_model,
               std::shared_ptr<torch::jit::script::Module> rl_model) {
  vector<ManeuverRecommendation *> recommendations;
  const auto road_users{database->findAll()};
  for (auto r : road_users) {
    if (r->getConnected() && r->getLanePosition() == 0) {
      auto neighbours = mapNeighbours(database, distanceRadius);
      auto input_values = RoadUsertoModelInput(r.get(), neighbours);
      auto models_input = torch::tensor(input_values).unsqueeze(0);
      recommendations.push_back(calculatedTrajectories(r.get(), models_input, lstm_model, rl_model));
      // auto models_input = torch::tensor(input_values).unsqueeze(0).unsqueeze(0);
      // if(!isCarTerminal(models_input)){
      // 	recommendations.push_back(calculatedTrajectories(r,models_input,lstm_model,rl_model));
      // } else {
      // 	r->setLanePosition(r->getLanePosition()+1);
      // }
    }
  }
  return recommendations;
}
