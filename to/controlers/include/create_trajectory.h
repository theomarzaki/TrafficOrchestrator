// This file uses both of the models(LSTM,RL) to compute way points for a 3 - window

// returns a maneuver recommendation to the TO

// Calculated actions to waypoints and vice versa

// change to n step window

// Created by: Omar Nassef (KCL)
#ifndef TO_CREATE_TRAJECTORY_H
#define TO_CREATE_TRAJECTORY_H

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>
#include <memory>
#include <time.h>

#include "road_user.h"
#include "nearest_neighbour.h"
#include "maneuver_recommendation.h"
#include "waypoint.h"
#include "mapper.h"
#include "database.h"

using namespace rapidjson;
using namespace std::chrono;
using namespace std;

bool inRange(int low, int high, int x);

at::Tensor GetStateFromActions(const at::Tensor &action_Tensor,const at::Tensor& state);

std::optional<std::pair<std::shared_ptr<RoadUser>,std::shared_ptr<RoadUser>>> getClosestFollowingandPreceedingCars(const std::shared_ptr<RoadUser>& merging_car, const std::vector<std::shared_ptr<RoadUser>>& close_by, int number_lanes_offset);

std::optional<vector<float>> RoadUsertoModelInput(const std::shared_ptr<RoadUser>& merging_car,
                                   const vector<pair<std::shared_ptr<RoadUser>,
                                   vector<std::shared_ptr<RoadUser>>>>& neighbours);

auto calculatedTrajectories(const std::shared_ptr<Database>& database,const std::shared_ptr<RoadUser>& mergingVehicle, const at::Tensor& models_input,const std::shared_ptr<torch::jit::script::Module>& rl_model) -> std::shared_ptr<ManeuverRecommendation>;

auto ManeuverParser(const std::shared_ptr<Database>& database,const std::shared_ptr<torch::jit::script::Module>& rl_model) -> vector<std::shared_ptr<ManeuverRecommendation>>;

#endif
