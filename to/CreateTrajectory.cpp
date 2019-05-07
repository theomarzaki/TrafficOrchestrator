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

double RoadUserGPStoProcessedGPS(int32_t point){
    //FIXME do not cast from a double to a float
	return double(point / pow(10,6));
}
double ProcessedGPStoRoadUserGPS(int32_t point){
    //FIXME do not cast from a double to a float
    return double(point * pow(10,6));
}

float RoadUserHeadingtoProcessedHeading(float point){
	return point / 100;
}

float ProcessedHeadingtoRoadUserHeading(float point){
	return point * 100;
}


bool inRange(int low, int high, int x){
    return ((x-high)*(x-low) <= 0);
}



auto getClosestFollowingandPreceedingCars(const std::shared_ptr<RoadUser> merging_car, std::vector<std::shared_ptr<RoadUser>> close_by) {
    std::shared_ptr<RoadUser> closest_following = nullptr;
    std::shared_ptr<RoadUser> closest_preceeding = nullptr;
    int minFollowing = 9999;
    int minPreceeding = 9999;

    for (const auto &close_car : close_by) {
        if (close_car->getLatitude() < merging_car->getLatitude() &&
            close_car->getLongitude() < merging_car->getLongitude() && close_car->getLanePosition() == merging_car->getLanePosition() + 1) { //closest following
            if (distanceEarth(RoadUserGPStoProcessedGPS(close_car->getLatitude()),
                              RoadUserGPStoProcessedGPS(close_car->getLongitude()),
                              RoadUserGPStoProcessedGPS(merging_car->getLatitude()),
                              RoadUserGPStoProcessedGPS(merging_car->getLongitude())) < minFollowing) {
                closest_following = close_car;
								minFollowing = distanceEarth(RoadUserGPStoProcessedGPS(close_car->getLatitude()),
		                              RoadUserGPStoProcessedGPS(close_car->getLongitude()),
		                              RoadUserGPStoProcessedGPS(merging_car->getLatitude()),
		                              RoadUserGPStoProcessedGPS(merging_car->getLongitude()));
            }
        }
        if (close_car->getLatitude() > merging_car->getLatitude() &&
            close_car->getLongitude() > merging_car->getLongitude() && close_car->getLanePosition() == merging_car->getLanePosition() + 1) { //closest preceeding
            if (distanceEarth(RoadUserGPStoProcessedGPS(close_car->getLatitude()),
                              RoadUserGPStoProcessedGPS(close_car->getLongitude()),
                              RoadUserGPStoProcessedGPS(merging_car->getLatitude()),
                              RoadUserGPStoProcessedGPS(merging_car->getLongitude())) < minPreceeding) {
                closest_preceeding = close_car;
								minPreceeding = distanceEarth(RoadUserGPStoProcessedGPS(close_car->getLatitude()),
		                              RoadUserGPStoProcessedGPS(close_car->getLongitude()),
		                              RoadUserGPStoProcessedGPS(merging_car->getLatitude()),
		                              RoadUserGPStoProcessedGPS(merging_car->getLongitude()));
            }
        }
    }
    if (closest_preceeding == nullptr) {
        //we create a default one
        closest_preceeding = std::make_shared<RoadUser>();
        // FIXME do not cast from a float (in reality a double) to int
        closest_preceeding->setLongitude(static_cast<int32_t>(RoadUserGPStoProcessedGPS(merging_car->getLongitude() + 10000))); // check
        // FIXME do not cast from a float (in reality a double) to int
        closest_preceeding->setLatitude(static_cast<int32_t>(RoadUserGPStoProcessedGPS(merging_car->getLatitude() + 10000))); //check
        closest_preceeding->setSpeed(merging_car->getSpeed());
        closest_preceeding->setWidth(merging_car->getWidth());
        closest_preceeding->setLength(merging_car->getLength());
        closest_preceeding->setAcceleration(merging_car->getAcceleration());
        closest_preceeding->setLanePosition(merging_car->getLanePosition() + 1);
    }
    if (closest_following == nullptr) {
        //we create a default one
        closest_following = std::make_shared<RoadUser>();
        // FIXME do not cast from a float (in reality a double) to int
        closest_following->setLongitude(static_cast<int32_t>(RoadUserGPStoProcessedGPS(merging_car->getLongitude() - 10000))); //check
        // FIXME do not cast from a float (in reality a double) to int
        closest_following->setLatitude(static_cast<int32_t>(RoadUserGPStoProcessedGPS(merging_car->getLatitude() - 10000))); // check
        closest_following->setSpeed(merging_car->getSpeed());
        closest_following->setWidth(merging_car->getWidth());
        closest_following->setLength(merging_car->getLength());
        closest_following->setAcceleration(merging_car->getAcceleration());
        closest_following->setLanePosition(merging_car->getLanePosition() + 1);
    }
    return std::make_pair(closest_preceeding, closest_following);
}


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
	} else perror("Action cannot be recognized");

	return stateTensor;
}

vector<float> RoadUsertoModelInput(const std::shared_ptr<RoadUser> merging_car,
                                   vector<pair<std::shared_ptr<RoadUser>,
                                   vector<std::shared_ptr<RoadUser>>>> neighbours) {

    std::pair<std::shared_ptr<RoadUser>, std::shared_ptr<RoadUser>> x;
    for (const auto &v : neighbours) {
        if (v.first->getUuid() == merging_car->getUuid()) {
            x = getClosestFollowingandPreceedingCars(merging_car, v.second);
        }
    }
    std::vector<float> mergingCar;
    mergingCar.push_back(RoadUserGPStoProcessedGPS(merging_car->getLatitude()));
    mergingCar.push_back(RoadUserGPStoProcessedGPS(merging_car->getLongitude()));
    mergingCar.push_back(merging_car->getLength());
    mergingCar.push_back(merging_car->getWidth());
    mergingCar.push_back(RoadUserSpeedtoProcessedSpeed(merging_car->getSpeed()));
    mergingCar.push_back(merging_car->getAcceleration());
    // FIXME do not cast from a double to a float

    mergingCar.push_back(distanceEarth(RoadUserGPStoProcessedGPS(merging_car->getLongitude()),RoadUserGPStoProcessedGPS(merging_car->getLatitude()),RoadUserGPStoProcessedGPS(x.first->getLongitude()),RoadUserGPStoProcessedGPS(x.first->getLatitude())));
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
		mergingCar.push_back(distanceEarth(RoadUserGPStoProcessedGPS(merging_car->getLongitude()),
                                                RoadUserGPStoProcessedGPS(merging_car->getLatitude()),
                                                RoadUserGPStoProcessedGPS(x.second->getLongitude()),
                                                RoadUserGPStoProcessedGPS(x.second->getLatitude())));
		mergingCar.push_back(RoadUserHeadingtoProcessedHeading(merging_car->getHeading()));
    return mergingCar;
}

auto calculatedTrajectories(Database * database,std::shared_ptr<RoadUser> mergingVehicle, at::Tensor models_input, std::shared_ptr<torch::jit::script::Module> lstm_model,
                            std::shared_ptr<torch::jit::script::Module> rl_model) {
    auto mergingManeuver{std::make_shared<ManeuverRecommendation>()};
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
    mergingManeuver->setMessageID(std::string(mergingManeuver->getOrigin()) + "/" + std::string(mergingManeuver->getUuidManeuver()) + "/" +
                                  std::string(to_string(mergingManeuver->getTimestamp())));

    rl_inputs.push_back(models_input);
    at::Tensor calculatedRL = rl_model->forward(rl_inputs).toTensor();
    auto calculated_n_1_states = GetStateFromActions(calculatedRL, models_input);

    auto waypoint{std::make_shared<Waypoint>()};
    waypoint->setTimestamp(timeCalculator.count() + (distanceEarth(mergingVehicle->getLatitude(), mergingVehicle->getLongitude(),
                                                                   calculated_n_1_states[0][0].item<float>(),
                                                                   calculated_n_1_states[0][1].item<float>()) /
                                                     mergingVehicle->getSpeed()) * 1000); //distance to mergeing point
    waypoint->setLatitude(ProcessedGPStoRoadUserGPS(calculated_n_1_states[0][0].item<float>()));
    waypoint->setLongitude(ProcessedGPStoRoadUserGPS(calculated_n_1_states[0][1].item<float>()));
    waypoint->setSpeed(ProcessedSpeedtoRoadUserSpeed(calculated_n_1_states[0][4].item<float>()));
    waypoint->setLanePosition(mergingVehicle->getLanePosition());
		waypoint->setHeading(ProcessedHeadingtoRoadUserHeading(calculated_n_1_states[0][19].item<float>()));
    mergingManeuver->addWaypoint(waypoint);
		mergingVehicle->setProcessingWaypoint(true);
		database->upsert(mergingVehicle);
    return mergingManeuver;
}

auto ManeuverParser(Database *database,
                    double distanceRadius,
                    std::shared_ptr<torch::jit::script::Module> lstm_model,
                    std::shared_ptr<torch::jit::script::Module> rl_model) {
    auto recommendations{vector<std::shared_ptr<ManeuverRecommendation>>()};
    const auto road_users{database->findAll()};
    for (const auto &r : road_users) {
        if (r->getConnected() && r->getLanePosition() == 0 && !(r->getProcessingWaypoint())) {
            auto neighbours{mapNeighbours(database, distanceRadius)};
            auto input_values{RoadUsertoModelInput(r, neighbours)};
            auto models_input{torch::tensor(input_values).unsqueeze(0)};
            recommendations.push_back(calculatedTrajectories(database,r, models_input, lstm_model, rl_model));
        }
    }
    return recommendations;

}
