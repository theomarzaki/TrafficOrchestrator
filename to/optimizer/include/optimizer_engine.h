//
// Created by jab on 27/05/19.
//

#ifndef TO_OPTIMIZER_ENGINE_H
#define TO_OPTIMIZER_ENGINE_H

#include <mapper.h>
#include <list>
#include <vector>
#include <memory>


class OptimizerEngine {
private:

    std::vector<Timebase_Telemetry_Waypoint> game;

public:

    OptimizerEngine() = default;

    bool updateSimulationState(std::unique_ptr<std::list<Timebase_Telemetry_Waypoint>>);
    std::unique_ptr<std::list<Timebase_Telemetry_Waypoint>> getSimulationResult(unsigned int delta_time);

};


#endif //TO_ENGINE_H
