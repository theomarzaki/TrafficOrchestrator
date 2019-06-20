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
#include "to/nearest_neighbour.cpp"
#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>
#include <memory>
#include <time.h>
#include "to/road_actions.cpp"
#include "mapper.cpp"

using namespace rapidjson;
using namespace std::chrono;


std::optional<std::pair<std::shared_ptr<RoadUser>,std::shared_ptr<RoadUser>>> getClosestFollowingandPreceedingCars(std::shared_ptr<RoadUser> merging_car, std::vector<std::shared_ptr<RoadUser>> close_by, int number_lanes_offset) {
    std::shared_ptr<RoadUser> closest_following{nullptr};
    std::shared_ptr<RoadUser> closest_preceeding{nullptr};
    double minFollowing{9999};
    double minPreceeding{9999};

    for (const auto &close_car : close_by) {
        if (close_car->getLatitude() < merging_car->getLatitude() &&
            close_car->getLongitude() < merging_car->getLongitude() && close_car->getLanePosition() == merging_car->getLanePosition() + number_lanes_offset) { //closest following
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
            close_car->getLongitude() > merging_car->getLongitude() && close_car->getLanePosition() == merging_car->getLanePosition() + number_lanes_offset) { //closest preceeding
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
        auto faker = Mapper::getMapper()->getFakeCarMergingScenario(toRealGPS(merging_car->getLatitude()),toRealGPS(merging_car->getLongitude()), 1);
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
	const int accelerate_tensor = 0;
	const int deccelerate_tensor = 1;
	const int left_tensor = 2;
	const int right_tensor = 3;
	const int doNothing_tensor = 4;

	switch(torch::argmax(action_Tensor).item<int>()){
		case accelerate_tensor:
			return Autonomous_action::accelerate(state);
		case deccelerate_tensor:
			return Autonomous_action::deccelerate(state);
		case left_tensor:
			return Autonomous_action::left(state);
		case right_tensor:
			return Autonomous_action::right(state);
		case doNothing_tensor:
			return Autonomous_action::nothing(state);
		default:
			logger::write("Action cannot be recognized");
      return state;
	}
}


std::optional<vector<float>> RoadUsertoModelInput(const std::shared_ptr<RoadUser> merging_car,
                                   vector<pair<std::shared_ptr<RoadUser>,
                                   vector<std::shared_ptr<RoadUser>>>> neighbours) {

    std::vector<std::shared_ptr<RoadUser>> no_neighbours;

    auto x = getClosestFollowingandPreceedingCars(merging_car,no_neighbours, 1); // get closest and following cars in the next lane ("target lane")


    for (const auto &v : neighbours) {
        if (v.first->getUuid() == merging_car->getUuid()) {
          x = getClosestFollowingandPreceedingCars(merging_car, v.second, 1);
        }
    }

		if (!x) {
				return std::nullopt;
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

auto calculatedTrajectories(const std::shared_ptr<Database> &database,
                            const std::shared_ptr<RoadUser> &mergingVehicle,
                            at::Tensor models_input,
                            const std::shared_ptr<torch::jit::script::Module> &rl_model) {
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

    rl_inputs.emplace_back(models_input);
    auto calculated_action = rl_model->forward(rl_inputs).toTensor();
    auto calculated_n_1 = GetStateFromActions(calculated_action, models_input);

    auto valid_action = CheckActionValidity(calculated_n_1);
    auto calculated_n_1_states = valid_action.first;

    auto waypoint{std::make_shared<Waypoint>()};
    waypoint->setTimestamp(timestamp + (distanceEarth(RoadUserGPStoProcessedGPS(mergingVehicle->getLatitude()), RoadUserGPStoProcessedGPS(mergingVehicle->getLongitude()),
                  calculated_n_1_states[0][0].item<float>(),calculated_n_1_states[0][1].item<float>()) / max(calculated_n_1_states[0][4].item<float>(),float(70))) * 100); //distance to mergeing point
    waypoint->setLongitude(ProcessedGPStoRoadUserGPS(calculated_n_1_states[0][0].item<float>()));
    waypoint->setLatitude(ProcessedGPStoRoadUserGPS(calculated_n_1_states[0][1].item<float>()));
    waypoint->setSpeed(ProcessedSpeedtoRoadUserSpeed(min(calculated_n_1_states[0][4].item<float>(),float(12))) / 3.6);
    (!(valid_action.second)) ? waypoint->setLanePosition(mergingVehicle->getLanePosition() + 1) : waypoint->setLanePosition(mergingVehicle->getLanePosition());
		waypoint->setHeading(ProcessedHeadingtoRoadUserHeading(calculated_n_1_states[0][19].item<float>()));
    mergingManeuver->addWaypoint(waypoint);
		mergingVehicle->setProcessingWaypoint(true);

		mergingVehicle->setWaypointTimeStamp(time(nullptr));

		database->upsert(mergingVehicle);
    return mergingManeuver;
}

auto ManeuverParser(std::shared_ptr<Database> database, std::shared_ptr<torch::jit::script::Module> rl_model) {
    auto recommendations{vector<std::shared_ptr<ManeuverRecommendation>>()};
    const auto road_users{database->findAll()};
    for (const auto &r : road_users) {
				if(difftime(time(nullptr),r->getWaypointTimestamp()) < 0){
					r->setProcessingWaypoint(false);
					database->upsert(r);
				}
					if (r->getConnected() && r->getLanePosition() == 0) { //&& !(r->getProcessingWaypoint()))
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
