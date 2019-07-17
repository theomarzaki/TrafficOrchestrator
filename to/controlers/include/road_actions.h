//
// Created by jab on 16/07/19.
//

#ifndef TO_ROAD_ACTIONS_H
#define TO_ROAD_ACTIONS_H

#include <maneuver_recommendation.h>
#include <waypoint.h>
#include <road_user.h>
#include <torch/torch.h>
#include <torch/script.h>


int RoadUserSpeedtoProcessedSpeed(int speed);
int ProcessedSpeedtoRoadUserSpeed(int speed);
double RoadUserGPStoProcessedGPS(float point);
double ProcessedGPStoRoadUserGPS(float point);

double toRealGPS(int32_t point);
int32_t toGPSMantissa(double point);

float RoadUserHeadingtoProcessedHeading(float point);
float ProcessedHeadingtoRoadUserHeading(float point);

bool inRange(int low, int high, int x);

std::pair<at::Tensor,bool> CheckActionValidity(at::Tensor states);

// Represents the actions that are taken by the non merging vehicle to facilitate an easier merge
namespace RoadUser_action {

    const float TIME_VARIANT = 0.035;
    const int BIAS = 1;

    auto left(std::shared_ptr<RoadUser> vehicle);
    auto right(std::shared_ptr<RoadUser> vehicle);
    auto accelerate(std::shared_ptr<RoadUser> vehicle);
    auto deccelerate(std::shared_ptr<RoadUser> vehicle);
    auto nothing(std::shared_ptr<RoadUser> vehicle);

}

// Represents the actions that is taken by the merging car -> used by RL
namespace Autonomous_action {

    const float TIME_VARIANT = 0.035;
    const int BIAS = 3;

    at::Tensor left(at::Tensor state);
    at::Tensor right(at::Tensor state);
    at::Tensor accelerate(at::Tensor state);
    at::Tensor deccelerate(at::Tensor state);
    at::Tensor nothing(at::Tensor state);

}

#endif //TO_ROAD_ACTIONS_H
