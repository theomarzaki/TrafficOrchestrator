// This file represents the actions that can be taken by the non merging vehicles on the road

using namespace std;

namespace RoadUser_action{

  auto left(RoadUser * vehicle){ 
    auto leftAction{std::make_shared<ManeuverRecommendation>()};
    return leftAction;
  }

  auto right(RoadUser * vehicle){
    auto rightAction{std::make_shared<ManeuverRecommendation>()};
    return rightAction;
  }

  auto accelerate(RoadUser * vehicle){
    auto accelerateAction{std::make_shared<ManeuverRecommendation>()};
    return accelerateAction;
  }

  auto deccelerate(RoadUser * vehicle){
    auto deccelerateAction{std::make_shared<ManeuverRecommendation>()};
    return deccelerateAction;
  }

  auto nothing(RoadUser * vehicle){
    auto nothinAction{std::make_shared<ManeuverRecommendation>()};
    return nothinAction;
  }

}

namespace Autonomous_action{

  const float TIME_VARIANT = 0.035;
  const int BIAS = 3;

  at::Tensor left(at::Tensor state){
    auto stateTensor = std::move(state);
    float displacement =  max(state[0][4].item<float>(),float(10)) * TIME_VARIANT + 0.5 * max(state[0][5].item<float>(),float(1)) * TIME_VARIANT * TIME_VARIANT;
    auto angle = (state[0][19].item<int>() + 1) % 360;
    auto new_x = state[0][0].item<float>() + BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = state[0][1].item<float>()  + BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);
    stateTensor[0][0] = new_x;
    stateTensor[0][1] = new_y;
    stateTensor[0][19] = angle;
    return stateTensor;
  }

  at::Tensor right(at::Tensor state){
    auto stateTensor = std::move(state);
    float displacement =  max(state[0][4].item<float>(),float(10)) * TIME_VARIANT + 0.5 * max(state[0][5].item<float>(),float(1)) * TIME_VARIANT * TIME_VARIANT;
    auto angle = (state[0][19].item<int>() + 1) % 360;
    auto new_x = state[0][0].item<float>() + BIAS * (displacement/1000) * cos((angle * M_PI)/ 180);
    auto new_y = state[0][1].item<float>()  + BIAS * (displacement/1000) * sin((angle * M_PI)/ 180);
    stateTensor[0][0] = new_x;
    stateTensor[0][1] = new_y;
    stateTensor[0][19] = angle;
    return stateTensor;
  }

  at::Tensor accelerate(at::Tensor state){
    auto stateTensor = std::move(state);
    auto angle = state[0][19].item<int>();
    auto final_velocity = max(state[0][4].item<float>(),float(10)) + TIME_VARIANT * (max(state[0][4].item<float>(),float(10)) + max(state[0][5].item<float>(),float(1)) * TIME_VARIANT);
    auto final_acceleration = (pow(final_velocity,2) - pow(max(state[0][4].item<float>(),float(10)),2)) / 2 * (0.5 * (max(state[0][4].item<float>(),float(10)) + final_velocity) * TIME_VARIANT);
    auto displacement = final_velocity * TIME_VARIANT + 0.5 * (final_acceleration * TIME_VARIANT * TIME_VARIANT);
    auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
    auto new_y = state[0][1].item<float>() + displacement * sin((angle * M_PI)/ 180);
    stateTensor[0][0] = new_x;
    stateTensor[0][1] = new_y;
    stateTensor[0][4] = final_velocity;
    stateTensor[0][5] = final_velocity;
    stateTensor[0][19] = angle;
    return stateTensor;
  }

  at::Tensor deccelerate(at::Tensor state){ //TODO check deccelerate Tensor
    auto stateTensor = std::move(state);
    auto angle = state[0][19].item<int>();
    auto final_velocity = max(state[0][4].item<float>(),float(10)) - TIME_VARIANT * (max(state[0][4].item<float>(),float(10)) + max(state[0][5].item<float>(),float(1)) * TIME_VARIANT);
		auto final_acceleration = (pow(final_velocity,2) - pow(max(state[0][4].item<float>(),float(10)),2)) / 2 * (0.5 * (max(state[0][4].item<float>(),float(10)) + final_velocity) * TIME_VARIANT);
		auto displacement = final_velocity * TIME_VARIANT + 0.5 * (final_acceleration * TIME_VARIANT * TIME_VARIANT);
		auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
		auto new_y = state[0][1].item<float>() + displacement * sin((angle * M_PI)/ 180);
		stateTensor[0][0] = new_x;
		stateTensor[0][1] = new_y;
		stateTensor[0][4] = final_velocity;
		stateTensor[0][5] = final_velocity;
		stateTensor[0][19] = angle;
	  return stateTensor;
  }

  at::Tensor nothing(at::Tensor state){
    auto stateTensor = std::move(state);
    auto angle = state[0][19].item<int>();
    auto displacement = max(state[0][4].item<float>(),float(10)) * TIME_VARIANT + 0.5 * (max(state[0][5].item<float>(),float(1)) * TIME_VARIANT * TIME_VARIANT);
    auto new_x = state[0][0].item<float>() + displacement * cos((angle * M_PI)/ 180);
    auto new_y = state[0][1].item<float>() + displacement * sin((angle * M_PI)/ 180);
    stateTensor[0][0] = new_x;
    stateTensor[0][1] = new_y;
    return stateTensor;
  }

}
