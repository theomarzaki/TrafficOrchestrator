// This file represents the actions that can be taken by the *connected* vehicles on the road

#include <chrono>
#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <time.h>


using namespace std;
using namespace std::chrono;

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


bool inRange(int low, int high, int x){
    return ((x-high)*(x-low) <= 0);
}

std::pair<at::Tensor,bool> CheckActionValidity(at::Tensor states){
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
		return std::make_pair(states,false);
	}

	return std::make_pair(states,true);
}

// Represents the actions that are taken by the non merging vehicle to facilitate an easier merge
namespace RoadUser_action{

  const float TIME_VARIANT = 0.035;
  const int BIAS = 1;

  auto left(std::shared_ptr<RoadUser> vehicle){
    auto timestamp{duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count()};
    auto waypoint{std::make_shared<Waypoint>()};


    float displacement = RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed()) * TIME_VARIANT + 0.5 * vehicle->getAcceleration() * TIME_VARIANT * TIME_VARIANT;
    auto angle = int(RoadUserHeadingtoProcessedHeading(vehicle->getHeading()) + 1) % 360;
    auto new_x = RoadUserGPStoProcessedGPS(vehicle->getLongitude()) + BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = RoadUserGPStoProcessedGPS(vehicle->getLatitude())  + BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);

    waypoint->setTimestamp(timestamp + (distanceEarth(RoadUserGPStoProcessedGPS(vehicle->getLatitude()), RoadUserGPStoProcessedGPS(vehicle->getLongitude()),
                  new_x,new_y) / RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed())) * 100);
    waypoint->setLongitude(ProcessedGPStoRoadUserGPS(new_x));
    waypoint->setLatitude(ProcessedGPStoRoadUserGPS(new_y));
    waypoint->setSpeed(vehicle->getSpeed());
    waypoint->setLanePosition(vehicle->getLanePosition());
		waypoint->setHeading(ProcessedHeadingtoRoadUserHeading(angle));

    return waypoint;
  }

  auto right(std::shared_ptr<RoadUser> vehicle){
    auto timestamp{duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count()};
    auto waypoint{std::make_shared<Waypoint>()};


    float displacement = RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed()) * TIME_VARIANT + 0.5 * vehicle->getAcceleration() * TIME_VARIANT * TIME_VARIANT;
    auto angle = int(RoadUserHeadingtoProcessedHeading(vehicle->getHeading()) - 1) % 360;
    auto new_x = RoadUserGPStoProcessedGPS(vehicle->getLongitude()) + BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = RoadUserGPStoProcessedGPS(vehicle->getLatitude())  + BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);

    waypoint->setTimestamp(timestamp + (distanceEarth(RoadUserGPStoProcessedGPS(vehicle->getLatitude()), RoadUserGPStoProcessedGPS(vehicle->getLongitude()),
                  new_x,new_y) / RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed())) * 100);
    waypoint->setLongitude(ProcessedGPStoRoadUserGPS(new_x));
    waypoint->setLatitude(ProcessedGPStoRoadUserGPS(new_y));
    waypoint->setSpeed(vehicle->getSpeed());
    waypoint->setLanePosition(vehicle->getLanePosition());
    waypoint->setHeading(ProcessedHeadingtoRoadUserHeading(angle));

    return waypoint;
  }

  auto accelerate(std::shared_ptr<RoadUser> vehicle){
    auto timestamp{duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count()};
    auto waypoint{std::make_shared<Waypoint>()};


    float displacement = RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed()) * TIME_VARIANT + 0.5 * vehicle->getAcceleration() * TIME_VARIANT * TIME_VARIANT;
    auto angle = int(RoadUserHeadingtoProcessedHeading(vehicle->getHeading())) % 360;
    auto new_x = RoadUserGPStoProcessedGPS(vehicle->getLongitude()) + BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = RoadUserGPStoProcessedGPS(vehicle->getLatitude())  + BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);

    waypoint->setTimestamp(timestamp + (distanceEarth(RoadUserGPStoProcessedGPS(vehicle->getLatitude()), RoadUserGPStoProcessedGPS(vehicle->getLongitude()),
                  new_x,new_y) / RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed())) * 100);
    waypoint->setLongitude(ProcessedGPStoRoadUserGPS(new_x));
    waypoint->setLatitude(ProcessedGPStoRoadUserGPS(new_y));
    waypoint->setSpeed(vehicle->getSpeed());
    waypoint->setLanePosition(vehicle->getLanePosition());
    waypoint->setHeading(ProcessedHeadingtoRoadUserHeading(angle));

    return waypoint;
  }

  auto deccelerate(std::shared_ptr<RoadUser> vehicle){
    auto timestamp{duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count()};
    auto waypoint{std::make_shared<Waypoint>()};


    float displacement = RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed()) * TIME_VARIANT + 0.5 * vehicle->getAcceleration() * TIME_VARIANT * TIME_VARIANT;
    auto angle = int(RoadUserHeadingtoProcessedHeading(vehicle->getHeading())) % 360;
    auto new_x = RoadUserGPStoProcessedGPS(vehicle->getLongitude()) + BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = RoadUserGPStoProcessedGPS(vehicle->getLatitude())  + BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);

    waypoint->setTimestamp(timestamp + (distanceEarth(RoadUserGPStoProcessedGPS(vehicle->getLatitude()), RoadUserGPStoProcessedGPS(vehicle->getLongitude()),
                  new_x,new_y) / RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed())) * 100);
    waypoint->setLongitude(ProcessedGPStoRoadUserGPS(new_x));
    waypoint->setLatitude(ProcessedGPStoRoadUserGPS(new_y));
    waypoint->setSpeed(vehicle->getSpeed());
    waypoint->setLanePosition(vehicle->getLanePosition());
    waypoint->setHeading(ProcessedHeadingtoRoadUserHeading(angle));

    return waypoint;
  }

  auto nothing(std::shared_ptr<RoadUser> vehicle){
    auto timestamp{duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count()};
    auto waypoint{std::make_shared<Waypoint>()};


    float displacement = RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed()) * TIME_VARIANT + 0.5 * vehicle->getAcceleration() * TIME_VARIANT * TIME_VARIANT;
    auto angle = int(RoadUserHeadingtoProcessedHeading(vehicle->getHeading())) % 360;
    auto new_x = RoadUserGPStoProcessedGPS(vehicle->getLongitude()) + BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = RoadUserGPStoProcessedGPS(vehicle->getLatitude())  + BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);

    waypoint->setTimestamp(timestamp + (distanceEarth(RoadUserGPStoProcessedGPS(vehicle->getLatitude()), RoadUserGPStoProcessedGPS(vehicle->getLongitude()),
                  new_x,new_y) / RoadUserSpeedtoProcessedSpeed(vehicle->getSpeed())) * 100);
    waypoint->setLongitude(ProcessedGPStoRoadUserGPS(new_x));
    waypoint->setLatitude(ProcessedGPStoRoadUserGPS(new_y));
    waypoint->setSpeed(vehicle->getSpeed());
    waypoint->setLanePosition(vehicle->getLanePosition());
    waypoint->setHeading(ProcessedHeadingtoRoadUserHeading(angle));

    return waypoint;
  }

}

// Represents the actions that is taken by the merging car -> used by RL
namespace Autonomous_action{

  const float TIME_VARIANT = 0.035;
  const int BIAS = 1;

  at::Tensor left(at::Tensor state){
    at::Tensor stateTensor = state;
    float displacement = max(state[0][4].item<float>(),float(10)) * TIME_VARIANT + 0.5 * max(state[0][5].item<float>(),float(1)) * TIME_VARIANT * TIME_VARIANT;
    auto angle = (state[0][19].item<int>() + 2) % 360;
    auto new_x = state[0][0].item<float>() + BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = state[0][1].item<float>() + BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);
    stateTensor[0][0] = new_x;
    stateTensor[0][1] = new_y;
    stateTensor[0][19] = angle;
    return stateTensor;
  }

  at::Tensor right(at::Tensor state){
    at::Tensor stateTensor = state;
    float displacement = max(state[0][4].item<float>(),float(10)) * TIME_VARIANT + 0.5 * max(state[0][5].item<float>(),float(1)) * TIME_VARIANT * TIME_VARIANT;
    auto angle = (state[0][19].item<int>() - 2) % 360;
    auto new_x = state[0][0].item<float>() + BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = state[0][1].item<float>() + BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);
    stateTensor[0][0] = new_x;
    stateTensor[0][1] = new_y;
    stateTensor[0][19] = angle;
    return stateTensor;
  }

  at::Tensor accelerate(at::Tensor state){
    at::Tensor stateTensor = state;
    auto angle = state[0][19].item<int>();
    auto final_velocity = max(state[0][4].item<float>(),float(10)) + TIME_VARIANT * (max(state[0][4].item<float>(),float(10)) + max(state[0][5].item<float>(),float(1)) * TIME_VARIANT);
    auto final_acceleration = (pow(final_velocity,2) - pow(max(state[0][4].item<float>(),float(10)),2)) / 2 * (0.5 * (max(state[0][4].item<float>(),float(10)) + final_velocity) * TIME_VARIANT);
    auto displacement = final_velocity * TIME_VARIANT + 0.5 * (final_acceleration * TIME_VARIANT * TIME_VARIANT);
    auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
    auto new_y = state[0][1].item<float>() + displacement * sin((angle * M_PI)/ 180);
    stateTensor[0][0] = new_x;
    stateTensor[0][1] = new_y;
    stateTensor[0][4] = final_velocity;
    stateTensor[0][5] = final_acceleration;
    stateTensor[0][19] = angle;
    return stateTensor;
  }

  at::Tensor deccelerate(at::Tensor state){ //TODO check deccelerate Tensor
    at::Tensor stateTensor = state;
    auto angle = state[0][19].item<int>();
    auto final_velocity = max(state[0][4].item<float>(),float(10)) - TIME_VARIANT * (max(state[0][4].item<float>(),float(10)) + max(state[0][5].item<float>(),float(1)) * TIME_VARIANT);
		auto final_acceleration = (pow(final_velocity,2) - pow(max(state[0][4].item<float>(),float(10)),2)) / 2 * (0.5 * (max(state[0][4].item<float>(),float(10)) + final_velocity) * TIME_VARIANT);
		auto displacement = final_velocity * TIME_VARIANT + 0.5 * (final_acceleration * TIME_VARIANT * TIME_VARIANT);
		auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
		auto new_y = state[0][1].item<float>() + displacement * sin((angle * M_PI)/ 180);
		stateTensor[0][0] = new_x;
		stateTensor[0][1] = new_y;
		stateTensor[0][4] = final_velocity;
		stateTensor[0][5] = final_acceleration;
		stateTensor[0][19] = angle;
	  return stateTensor;
  }

  at::Tensor nothing(at::Tensor state){
    at::Tensor stateTensor = state;
    auto angle = state[0][19].item<int>();
    auto displacement = max(state[0][4].item<float>(),float(10)) * TIME_VARIANT + 0.5 * (max(state[0][5].item<float>(),float(1)) * TIME_VARIANT * TIME_VARIANT);
    auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
    auto new_y = state[0][1].item<float>() + displacement * sin((angle * M_PI)/ 180);
    stateTensor[0][0] = new_x;
    stateTensor[0][1] = new_y;
    return stateTensor;
  }

}
