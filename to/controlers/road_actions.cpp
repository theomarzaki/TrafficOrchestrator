// This file represents the actions that can be taken by the *connected* vehicles on the road

#include "road_actions.h"

#include <chrono>
#include <logger.h>
#include <memory>
#include <time.h>
#include <nearest_neighbour.h>

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
	return float(int((point / 100) + 265) % 360);
}

float ProcessedHeadingtoRoadUserHeading(float point){
	return float(int((point * 100) - 265) % 360);
}

std::pair<at::Tensor,bool> CheckActionValidity(const at::Tensor& states){
	float THRESHOLD = 1400;

	auto x = states[0][0].item<float>();
	auto y = states[0][1].item<float>();

	auto pre_x = states[0][7].item<float>();
	auto pre_y = states[0][8].item<float>();

	auto fol_x = states[0][13].item<float>();
	auto fol_y = states[0][14].item<float>();

	auto slope =  (fol_y - pre_y) / (fol_x - pre_x);
	auto plus_c = pre_y - (slope * pre_x);


	if(y - (x * slope + plus_c) < THRESHOLD){
		logger::write("DEBUG :: Car Sucessfully Merged");
		return {states,false};
	}

	return {states,true};
}


  auto RoadUser_action::left(const std::shared_ptr<RoadUser>& vehicle) {
    auto timestamp{std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()};
    auto waypoint{std::make_shared<Waypoint>()};


    float displacement = RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed()) * RoadUser_action::TIME_VARIANT + 0.5 * vehicle->getAcceleration() * RoadUser_action::TIME_VARIANT * RoadUser_action::TIME_VARIANT;
    auto angle = int(RoadUserHeadingtoProcessedHeading(vehicle->getHeading()) + 1) % 360;
    auto new_x = RoadUserGPStoProcessedGPS(vehicle->getLongitude()) + RoadUser_action::BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = RoadUserGPStoProcessedGPS(vehicle->getLatitude())  + RoadUser_action::BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);

    waypoint->setTimestamp(static_cast<uint64_t>(timestamp + (distanceEarth(RoadUserGPStoProcessedGPS(vehicle->getLatitude()), RoadUserGPStoProcessedGPS(vehicle->getLongitude()),
                  new_x,new_y) / RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed())) * 100));
    waypoint->setLongitude(static_cast<uint32_t>(ProcessedGPStoRoadUserGPS(new_x)));
    waypoint->setLatitude(static_cast<uint32_t>(ProcessedGPStoRoadUserGPS(new_y)));
    waypoint->setSpeed(vehicle->getSpeed());
    waypoint->setLanePosition(vehicle->getLanePosition());
		waypoint->setHeading(static_cast<uint16_t>(ProcessedHeadingtoRoadUserHeading(angle)));

    return waypoint;
  }

  auto RoadUser_action::right(const std::shared_ptr<RoadUser>& vehicle) {
    auto timestamp{std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()};
    auto waypoint{std::make_shared<Waypoint>()};


    float displacement = RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed()) * RoadUser_action::TIME_VARIANT + 0.5 * vehicle->getAcceleration() * RoadUser_action::TIME_VARIANT * RoadUser_action::TIME_VARIANT;
    auto angle = int(RoadUserHeadingtoProcessedHeading(vehicle->getHeading()) - 1) % 360;
    auto new_x = RoadUserGPStoProcessedGPS(vehicle->getLongitude()) + RoadUser_action::BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = RoadUserGPStoProcessedGPS(vehicle->getLatitude())  + RoadUser_action::BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);

    waypoint->setTimestamp(static_cast<uint64_t>(timestamp + (distanceEarth(RoadUserGPStoProcessedGPS(vehicle->getLatitude()), RoadUserGPStoProcessedGPS(vehicle->getLongitude()),
                  new_x,new_y) / RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed())) * 100));
    waypoint->setLongitude(static_cast<uint32_t>(ProcessedGPStoRoadUserGPS(new_x)));
    waypoint->setLatitude(static_cast<uint32_t>(ProcessedGPStoRoadUserGPS(new_y)));
    waypoint->setSpeed(vehicle->getSpeed());
    waypoint->setLanePosition(vehicle->getLanePosition());
    waypoint->setHeading(static_cast<uint16_t>(ProcessedHeadingtoRoadUserHeading(angle)));

    return waypoint;
  }

  auto RoadUser_action::accelerate(const std::shared_ptr<RoadUser>& vehicle) {
    auto timestamp{std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()};
    auto waypoint{std::make_shared<Waypoint>()};


    float displacement = RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed()) * RoadUser_action::TIME_VARIANT + 0.5 * vehicle->getAcceleration() * RoadUser_action::TIME_VARIANT * RoadUser_action::TIME_VARIANT;
    auto angle = int(RoadUserHeadingtoProcessedHeading(vehicle->getHeading())) % 360;
    auto new_x = RoadUserGPStoProcessedGPS(vehicle->getLongitude()) + RoadUser_action::BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = RoadUserGPStoProcessedGPS(vehicle->getLatitude())  + RoadUser_action::BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);

    waypoint->setTimestamp(static_cast<uint64_t>(timestamp + (distanceEarth(RoadUserGPStoProcessedGPS(vehicle->getLatitude()), RoadUserGPStoProcessedGPS(vehicle->getLongitude()),
                  new_x,new_y) / RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed())) * 100));
    waypoint->setLongitude(static_cast<uint32_t>(ProcessedGPStoRoadUserGPS(new_x)));
    waypoint->setLatitude(static_cast<uint32_t>(ProcessedGPStoRoadUserGPS(new_y)));
    waypoint->setSpeed(vehicle->getSpeed());
    waypoint->setLanePosition(vehicle->getLanePosition());
    waypoint->setHeading(static_cast<uint16_t>(ProcessedHeadingtoRoadUserHeading(angle)));

    return waypoint;
  }

  auto RoadUser_action::deccelerate(const std::shared_ptr<RoadUser>& vehicle) {
    auto timestamp{std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()};
    auto waypoint{std::make_shared<Waypoint>()};


    float displacement = RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed()) * RoadUser_action::TIME_VARIANT + 0.5 * vehicle->getAcceleration() * RoadUser_action::TIME_VARIANT * RoadUser_action::TIME_VARIANT;
    auto angle = int(RoadUserHeadingtoProcessedHeading(vehicle->getHeading())) % 360;
    auto new_x = RoadUserGPStoProcessedGPS(vehicle->getLongitude()) + RoadUser_action::BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = RoadUserGPStoProcessedGPS(vehicle->getLatitude())  + RoadUser_action::BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);

    waypoint->setTimestamp(static_cast<uint64_t>(timestamp + (distanceEarth(RoadUserGPStoProcessedGPS(vehicle->getLatitude()), RoadUserGPStoProcessedGPS(vehicle->getLongitude()),
                  new_x,new_y) / RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed())) * 100));
    waypoint->setLongitude(static_cast<uint32_t>(ProcessedGPStoRoadUserGPS(new_x)));
    waypoint->setLatitude(static_cast<uint32_t>(ProcessedGPStoRoadUserGPS(new_y)));
    waypoint->setSpeed(vehicle->getSpeed());
    waypoint->setLanePosition(vehicle->getLanePosition());
    waypoint->setHeading(static_cast<uint16_t>(ProcessedHeadingtoRoadUserHeading(angle)));

    return waypoint;
  }

  auto RoadUser_action::nothing(const std::shared_ptr<RoadUser>& vehicle) {
    auto timestamp{std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()};
    auto waypoint{std::make_shared<Waypoint>()};


    float displacement = RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed()) * RoadUser_action::TIME_VARIANT + 0.5 * vehicle->getAcceleration() * RoadUser_action::TIME_VARIANT * RoadUser_action::TIME_VARIANT;
    auto angle = int(RoadUserHeadingtoProcessedHeading(vehicle->getHeading())) % 360;
    auto new_x = RoadUserGPStoProcessedGPS(vehicle->getLongitude()) + RoadUser_action::BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = RoadUserGPStoProcessedGPS(vehicle->getLatitude())  + RoadUser_action::BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);

    waypoint->setTimestamp(static_cast<uint64_t>(timestamp + (distanceEarth(RoadUserGPStoProcessedGPS(vehicle->getLatitude()), RoadUserGPStoProcessedGPS(vehicle->getLongitude()),
                  new_x,new_y) / RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed())) * 100));
    waypoint->setLongitude(static_cast<uint32_t>(ProcessedGPStoRoadUserGPS(new_x)));
    waypoint->setLatitude(static_cast<uint32_t>(ProcessedGPStoRoadUserGPS(new_y)));
    waypoint->setSpeed(vehicle->getSpeed());
    waypoint->setLanePosition(vehicle->getLanePosition());
    waypoint->setHeading(static_cast<uint16_t>(ProcessedHeadingtoRoadUserHeading(angle)));

    return waypoint;
}


  at::Tensor Autonomous_action::left(const at::Tensor& state) {
    at::Tensor stateTensor = state;
    float displacement = fmax(state[0][4].item<float>(),float(10)) * Autonomous_action::TIME_VARIANT + 0.5 * fmax(state[0][5].item<float>(),float(1)) * Autonomous_action::TIME_VARIANT * Autonomous_action::TIME_VARIANT;
    auto angle = (state[0][19].item<int>() + 2) % 360;
    auto new_x = state[0][0].item<float>() + Autonomous_action::BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = state[0][1].item<float>() + Autonomous_action::BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);
    stateTensor[0][0] = new_x;
    stateTensor[0][1] = new_y;
    stateTensor[0][19] = angle;
    return stateTensor;
  }

  at::Tensor Autonomous_action::right(const at::Tensor& state) {
    at::Tensor stateTensor = state;
    float displacement = fmax(state[0][4].item<float>(),float(10)) * Autonomous_action::TIME_VARIANT + 0.5 * fmax(state[0][5].item<float>(),float(1)) * Autonomous_action::TIME_VARIANT * Autonomous_action::TIME_VARIANT;
    auto angle = (state[0][19].item<int>() - 2) % 360;
    auto new_x = state[0][0].item<float>() + Autonomous_action::BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = state[0][1].item<float>() + Autonomous_action::BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);
    stateTensor[0][0] = new_x;
    stateTensor[0][1] = new_y;
    stateTensor[0][19] = angle;
    return stateTensor;
  }

  at::Tensor Autonomous_action::accelerate(const at::Tensor& state) {
    at::Tensor stateTensor = state;
    auto angle = state[0][19].item<int>();
    auto final_velocity = fmax(state[0][4].item<float>(),float(10)) + Autonomous_action::TIME_VARIANT * (fmax(state[0][4].item<float>(),float(10)) + fmax(state[0][5].item<float>(),float(1)) * Autonomous_action::TIME_VARIANT);
    auto final_acceleration = (pow(final_velocity,2) - pow(fmax(state[0][4].item<float>(),float(10)),2)) / 2 * (0.5 * (fmax(state[0][4].item<float>(),float(10)) + final_velocity) * Autonomous_action::TIME_VARIANT);
    auto displacement = final_velocity * Autonomous_action::TIME_VARIANT + 0.5 * (final_acceleration * Autonomous_action::TIME_VARIANT * Autonomous_action::TIME_VARIANT);
    auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
    auto new_y = state[0][1].item<float>() + displacement * sin((angle * M_PI)/ 180);
    stateTensor[0][0] = new_x;
    stateTensor[0][1] = new_y;
    stateTensor[0][4] = final_velocity;
    stateTensor[0][5] = final_acceleration;
    stateTensor[0][19] = angle;
    return stateTensor;
  }

  at::Tensor Autonomous_action::deccelerate(const at::Tensor& state) { //TODO check deccelerate Tensor
    at::Tensor stateTensor = state;
    auto angle = state[0][19].item<int>();
    auto final_velocity = fmax(state[0][4].item<float>(),float(10)) - Autonomous_action::TIME_VARIANT * (fmax(state[0][4].item<float>(),float(10)) + fmax(state[0][5].item<float>(),float(1)) * Autonomous_action::TIME_VARIANT);
		auto final_acceleration = (pow(final_velocity,2) - pow(fmax(state[0][4].item<float>(),float(10)),2)) / 2 * (0.5 * (fmax(state[0][4].item<float>(),float(10)) + final_velocity) * Autonomous_action::TIME_VARIANT);
		auto displacement = final_velocity * Autonomous_action::TIME_VARIANT + 0.5 * (final_acceleration * Autonomous_action::TIME_VARIANT * Autonomous_action::TIME_VARIANT);
		auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
		auto new_y = state[0][1].item<float>() + displacement * sin((angle * M_PI)/ 180);
		stateTensor[0][0] = new_x;
		stateTensor[0][1] = new_y;
		stateTensor[0][4] = final_velocity;
		stateTensor[0][5] = final_acceleration;
		stateTensor[0][19] = angle;
	  return stateTensor;
  }

  at::Tensor Autonomous_action::nothing(const at::Tensor& state) {
    at::Tensor stateTensor = state;
    auto angle = state[0][19].item<int>();
    auto displacement = fmax(state[0][4].item<float>(),float(10)) * Autonomous_action::TIME_VARIANT + 0.5 * (fmax(state[0][5].item<float>(),float(1)) * Autonomous_action::TIME_VARIANT * Autonomous_action::TIME_VARIANT);
    auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
    auto new_y = state[0][1].item<float>() + displacement * sin((angle * M_PI)/ 180);
    stateTensor[0][0] = new_x;
    stateTensor[0][1] = new_y;
    return stateTensor;
  }