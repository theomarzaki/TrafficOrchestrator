//
// Created by jab on 27/05/19.
//

#ifndef TO_OPTIMIZER_ENGINE_H
#define TO_OPTIMIZER_ENGINE_H

#define INTERVAL_TIME 200

#include <list>
#include <map>
#include <memory>
#include <future>

#include <network_interface.h>
#include <maneuver_recommendation.h>
#include <road_user.h>
#include <mapper.h>


class OptimizerEngine {
private:
    inline static std::shared_ptr<OptimizerEngine> engine;

    std::map<std::string,Timebase_Telemetry_Waypoint> game;
    std::atomic_bool kill;
    std::atomic_bool pause;
    std::shared_ptr<std::thread> optimizerT;

    OptimizerEngine();

    void setBatch(size_t interval);

public:

    static Timebase_Telemetry_Waypoint createTelemetryElementFromRoadUser(const std::shared_ptr<RoadUser>& car);
    static std::shared_ptr<ManeuverRecommendation> telemetryStructToManeuverRecommendation(Timebase_Telemetry_Waypoint car);
    static std::shared_ptr<OptimizerEngine> getEngine();

    std::shared_ptr<std::thread> getThread();
    void killOptimizer();
    void startManeuverFeedback();
    void pauseManeuverFeedback();
    void updateSimulationState(std::unique_ptr<std::list<std::shared_ptr<RoadUser>>> cars);
    std::unique_ptr<std::list<Timebase_Telemetry_Waypoint>> getSimulationResult();

};

#endif
