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
#include <time.h>

#include "mapper.cpp"

using namespace std;
using namespace rapidjson;

float TIME_VARIANT = 0.035;
int BIAS = 3;

using namespace std::chrono;
using std::cout;
time_t waypointTimeCalculator;

int RoadUserSpeedtoProcessedSpeed(int speed){
	return speed / 100;
}

int ProcessedSpeedtoRoadUserSpeed(int speed){
	return speed * 100;
}

double RoadUserGPStoProcessedGPS(float point){
    //FIXME do not cast from a double to a float
	return double(point / pow(10,6));
}
double ProcessedGPStoRoadUserGPS(float point){
    //FIXME do not cast from a double to a float
    return double(point * pow(10,6));
}

double toRealGPS(int32_t point){
    return point / pow(10,7);
}
int32_t toGPSMantissa(double point){
    return point * pow(10,7);
}

float RoadUserHeadingtoProcessedHeading(float point){
	return float(int((point / 100) + 270) % 360);
}

float ProcessedHeadingtoRoadUserHeading(float point){
	return float(int((point * 100) - 270) % 360);
}


bool inRange(int low, int high, int x){
    return ((x-high)*(x-low) <= 0);
}



std::optional<std::pair<std::shared_ptr<RoadUser>,std::shared_ptr<RoadUser>>> getClosestFollowingandPreceedingCars(std::shared_ptr<RoadUser> merging_car, std::vector<std::shared_ptr<RoadUser>> close_by) {
    std::shared_ptr<RoadUser> closest_following{nullptr};
    std::shared_ptr<RoadUser> closest_preceeding{nullptr};
    double minFollowing{9999};
    double minPreceeding{9999};

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
		// closest_following = nullptr;
		// closest_preceeding = nullptr;
    if (closest_preceeding == nullptr or closest_following == nullptr ) { // Calculate once only
        auto faker = Mapper::getMapper()->getFakeCarMergingScenario(toRealGPS(merging_car->getLatitude()),toRealGPS(merging_car->getLongitude()));
        if (faker) {
            if (closest_preceeding == nullptr) {
                //we create a default one
                closest_preceeding = std::make_shared<RoadUser>();
                closest_preceeding->setLongitude(toGPSMantissa(faker->preceeding.longitude));
                closest_preceeding->setLatitude(toGPSMantissa(faker->preceeding.latitude));
                closest_preceeding->setSpeed(merging_car->getSpeed());
                closest_preceeding->setWidth(merging_car->getWidth());
                closest_preceeding->setLength(merging_car->getLength());
                closest_preceeding->setAcceleration(merging_car->getAcceleration());
                closest_preceeding->setLanePosition(merging_car->getLanePosition() + 1);
            }
            if (closest_following == nullptr) {
                //we create a default one
                closest_following = std::make_shared<RoadUser>();
                closest_following->setLongitude(toGPSMantissa(faker->following.longitude));
                closest_following->setLatitude(toGPSMantissa(faker->following.latitude));
                closest_following->setSpeed(merging_car->getSpeed());
                closest_following->setWidth(merging_car->getWidth());
                closest_following->setLength(merging_car->getLength());
                closest_following->setAcceleration(merging_car->getAcceleration());
                closest_following->setLanePosition(merging_car->getLanePosition() + 1);
            }
        } else {
            return std::nullopt;
        }
    }
    return std::make_pair(closest_preceeding, closest_following);
}


at::Tensor GetStateFromActions(const at::Tensor &action_Tensor,at::Tensor state){
	int accelerate_tensor = 0;
	int deccelerate_tensor = 1;
	int left_tensor = 2;
	int right_tensor = 3;
	int doNothing_tensor = 4;

	auto stateTensor = state;

	auto merging_Long = state[0][0].item<float>();
	auto merging_Lat = state[0][1].item<float>();
	auto merging_Speed = max(state[0][4].item<float>(),float(10));
	auto merging_Acc = max(state[0][5].item<float>(),float(1));
	auto angle = state[0][19].item<int>();


	auto actionTensor = torch::argmax(action_Tensor);
	if(accelerate_tensor == actionTensor.item<int>()){
			auto final_velocity = merging_Speed + TIME_VARIANT * (merging_Speed + merging_Acc * TIME_VARIANT);
			auto final_acceleration = (pow(final_velocity,2) - pow(merging_Speed,2)) / 2 * (0.5 * (merging_Speed + final_velocity) * TIME_VARIANT);
			auto displacement = final_velocity * TIME_VARIANT + 0.5 * (final_acceleration * TIME_VARIANT * TIME_VARIANT);
			auto new_x = merging_Long + displacement * cos((angle * M_PI)/ 180);
			auto new_y = merging_Lat + displacement * sin((angle * M_PI)/ 180);
			stateTensor[0][0] = new_x;
			stateTensor[0][1] = new_y;
			stateTensor[0][4] = final_velocity;
			stateTensor[0][5] = final_velocity;
			stateTensor[0][19] = angle;
		return stateTensor;
	} else if(deccelerate_tensor == actionTensor.item<int>()){
		auto final_velocity = merging_Speed - TIME_VARIANT * (merging_Speed + merging_Acc * TIME_VARIANT);
		auto final_acceleration = (pow(final_velocity,2) - pow(merging_Speed,2)) / 2 * (0.5 * (merging_Speed + final_velocity) * TIME_VARIANT);
		auto displacement = final_velocity * TIME_VARIANT + 0.5 * (final_acceleration * TIME_VARIANT * TIME_VARIANT);
		auto new_x = merging_Long + displacement * cos((angle * M_PI)/ 180);
		auto new_y = merging_Lat + displacement * sin((angle * M_PI)/ 180);
		stateTensor[0][0] = new_x;
		stateTensor[0][1] = new_y;
		stateTensor[0][4] = final_velocity;
		stateTensor[0][5] = final_velocity;
		stateTensor[0][19] = angle;
	return stateTensor;
	} else if(left_tensor == actionTensor.item<int>()){
			float displacement = merging_Speed * TIME_VARIANT + 0.5 * merging_Acc * TIME_VARIANT * TIME_VARIANT;
			angle = (angle + 1) % 360;
			auto new_x = merging_Long + BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
			auto new_y = merging_Lat  + BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);
			stateTensor[0][0] = new_x;
			stateTensor[0][1] = new_y;
			stateTensor[0][19] = angle;
		return stateTensor;
	} else if(right_tensor == actionTensor.item<int>()){
			float displacement = merging_Speed * TIME_VARIANT + 0.5 * merging_Acc * TIME_VARIANT * TIME_VARIANT;
			angle = (angle - 1) % 360;
			auto new_x = merging_Long + BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
			auto new_y = merging_Lat  + BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);
			stateTensor[0][0] = new_x;
			stateTensor[0][1] = new_y;
			stateTensor[0][19] = angle;
		return stateTensor;
	} else if(doNothing_tensor == actionTensor.item<int>()){
			auto displacement = merging_Speed * TIME_VARIANT + 0.5 * (merging_Acc * TIME_VARIANT * TIME_VARIANT);
	    auto new_x = merging_Long + displacement * cos((angle * M_PI)/ 180);
	    auto new_y = merging_Lat + displacement * sin((angle * M_PI)/ 180);
	    stateTensor[0][0] = new_x;
	    stateTensor[0][1] = new_y;
			stateTensor[0][19] = angle;
    return stateTensor;
	} else perror("Action cannot be recognized");

	return stateTensor;
}

std::optional<vector<float>> RoadUsertoModelInput(const std::shared_ptr<RoadUser> merging_car,
                                   vector<pair<std::shared_ptr<RoadUser>,
                                   vector<std::shared_ptr<RoadUser>>>> neighbours) {

    std::vector<std::shared_ptr<RoadUser>> no_neighbours;

    auto x = getClosestFollowingandPreceedingCars(merging_car,no_neighbours);
    if (!x) {
        return std::nullopt;
    }

    for (const auto &v : neighbours) {
        if (v.first->getUuid() == merging_car->getUuid()) {
            x = getClosestFollowingandPreceedingCars(merging_car, v.second);
        }
    }


    std::vector<float> mergingCar;
    mergingCar.push_back(RoadUserGPStoProcessedGPS(merging_car->getLongitude()));
    mergingCar.push_back(RoadUserGPStoProcessedGPS(merging_car->getLatitude()));
    mergingCar.push_back(merging_car->getLength());
    mergingCar.push_back(merging_car->getWidth());
    mergingCar.push_back(RoadUserSpeedtoProcessedSpeed(merging_car->getSpeed()));
    mergingCar.push_back(merging_car->getAcceleration());
    // FIXME do not cast from a double to a float
    mergingCar.push_back(distanceEarth(RoadUserGPStoProcessedGPS(merging_car->getLongitude()),RoadUserGPStoProcessedGPS(merging_car->getLatitude()),RoadUserGPStoProcessedGPS(x->first->getLongitude()),RoadUserGPStoProcessedGPS(x->first->getLatitude())));
		mergingCar.push_back(RoadUserGPStoProcessedGPS(x->first->getLatitude()));
    mergingCar.push_back(RoadUserGPStoProcessedGPS(x->first->getLongitude()));
    mergingCar.push_back(x->first->getLength());
    mergingCar.push_back(x->first->getWidth());
    mergingCar.push_back(RoadUserSpeedtoProcessedSpeed(x->first->getSpeed()));
    mergingCar.push_back(x->first->getAcceleration());
    mergingCar.push_back(RoadUserGPStoProcessedGPS(x->second->getLatitude()));
    mergingCar.push_back(RoadUserGPStoProcessedGPS(x->second->getLongitude()));
    mergingCar.push_back(x->second->getWidth());
    mergingCar.push_back(RoadUserSpeedtoProcessedSpeed(x->second->getSpeed()));
    mergingCar.push_back(x->second->getAcceleration());
		mergingCar.push_back(distanceEarth(RoadUserGPStoProcessedGPS(merging_car->getLongitude()),
                                                RoadUserGPStoProcessedGPS(merging_car->getLatitude()),
                                                RoadUserGPStoProcessedGPS(x->second->getLongitude()),
                                                RoadUserGPStoProcessedGPS(x->second->getLatitude())));
		mergingCar.push_back(RoadUserHeadingtoProcessedHeading(merging_car->getHeading()));
    return mergingCar;
}

auto calculatedTrajectories(Database * database,std::shared_ptr<RoadUser> mergingVehicle, at::Tensor models_input,std::shared_ptr<torch::jit::script::Module> rl_model) {
    auto mergingManeuver{std::make_shared<ManeuverRecommendation>()};
    std::vector<torch::jit::IValue> rl_inputs;

		auto timestamp = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

    mergingManeuver->setTimestamp(timestamp);
    mergingManeuver->setUuidVehicle(mergingVehicle->getUuid());
    mergingManeuver->setUuidTo(mergingVehicle->getUuid());
    mergingManeuver->setTimestampAction(timestamp);
    mergingManeuver->setLongitudeAction(mergingVehicle->getLongitude());
    mergingManeuver->setLatitudeAction(mergingVehicle->getLatitude());
    mergingManeuver->setSpeedAction(ProcessedSpeedtoRoadUserSpeed(mergingVehicle->getSpeed()));
    mergingManeuver->setLanePositionAction(mergingVehicle->getLanePosition());
    mergingManeuver->setMessageID(std::string(mergingManeuver->getOrigin()) + "/" + std::string(mergingManeuver->getUuidManeuver()) + "/" +
                                  std::string(to_string(mergingManeuver->getTimestamp())));

    rl_inputs.push_back(models_input);
    at::Tensor calculatedRL = rl_model->forward(rl_inputs).toTensor();
    auto calculated_n_1_states = GetStateFromActions(calculatedRL, models_input);

		auto merging_Speed = max(calculated_n_1_states[0][4].item<float>(),float(10));

    auto waypoint{std::make_shared<Waypoint>()};
    waypoint->setTimestamp(timestamp + (distanceEarth(RoadUserGPStoProcessedGPS(mergingVehicle->getLatitude()), RoadUserGPStoProcessedGPS(mergingVehicle->getLongitude()),
                  calculated_n_1_states[0][0].item<float>(),calculated_n_1_states[0][1].item<float>()) / merging_Speed) * 100); //distance to mergeing point
    waypoint->setLongitude(ProcessedGPStoRoadUserGPS(calculated_n_1_states[0][0].item<float>()));
    waypoint->setLatitude(ProcessedGPStoRoadUserGPS(calculated_n_1_states[0][1].item<float>()));
    waypoint->setSpeed(ProcessedSpeedtoRoadUserSpeed(merging_Speed));
    waypoint->setLanePosition(mergingVehicle->getLanePosition());
		waypoint->setHeading(ProcessedHeadingtoRoadUserHeading(calculated_n_1_states[0][19].item<float>()));
    mergingManeuver->addWaypoint(waypoint);
		mergingVehicle->setProcessingWaypoint(true);

		mergingVehicle->setWaypointTimeStamp(time(NULL));

		database->upsert(mergingVehicle);
    return mergingManeuver;
}

auto ManeuverParser(Database *database,std::shared_ptr<torch::jit::script::Module> rl_model) {
    auto recommendations{vector<std::shared_ptr<ManeuverRecommendation>>()};
    const auto road_users{database->findAll()};
    for (const auto &r : road_users) {
				if(difftime(time(NULL),r->getWaypointTimestamp()) < 0){
					r->setProcessingWaypoint(false);
					database->upsert(r);
				}
					if (r->getConnected() && r->getLanePosition() == 0 && !(r->getProcessingWaypoint())) {
	            auto neighbours{mapNeighbours(database, 10000)};
	            auto input_values{RoadUsertoModelInput(r, neighbours)};
							if (input_values) {
	                auto models_input{torch::tensor(input_values.value()).unsqueeze(0)};
	                recommendations.push_back(calculatedTrajectories(database,r, models_input, rl_model));
	            }
	        }
    }
    return recommendations;
}
