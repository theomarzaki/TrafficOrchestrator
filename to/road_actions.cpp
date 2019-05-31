// This file represents the actions that can be taken by the non merging vehicles on the road

using namespace std;
namespace RoadUser_action{

  RoadUser * left(RoadUser * vehicle){
    cout << "here" << endl;
    return vehicle;
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

}
