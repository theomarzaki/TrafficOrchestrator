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
#include <condition_variable>

#include <maneuver_recommendation.h>
#include <road_user.h>
#include <gpstools.h>


class OptimizerEngine {
private:
    inline static OptimizerEngine* engine;

    std::map<std::string,Timebase_Telemetry_Waypoint> game;
    std::atomic_bool kill{};
    std::shared_ptr<std::thread> optimizerT;
    std::condition_variable cv;

    std::mutex pause;
    bool fence{false};

    OptimizerEngine();

    void setBatch(size_t interval);

    struct Graph_Element {
        std::shared_ptr<Timebase_Telemetry_Waypoint> telemetry;
        std::list<std::shared_ptr<Graph_Element>> in_front_neighbours;
        std::list<std::shared_ptr<Graph_Element>> behind_neighbours;
    };

public:
    std::mutex locker;

    static Timebase_Telemetry_Waypoint forceCarMerging(Timebase_Telemetry_Waypoint car, int64_t interval, int64_t timenow);
    static Timebase_Telemetry_Waypoint getPositionOnRoadInInterval(Timebase_Telemetry_Waypoint car, int64_t interval, int64_t timenow);
    static Timebase_Telemetry_Waypoint createTelemetryElementFromRoadUser(const std::shared_ptr<RoadUser>& car);
    static std::shared_ptr<ManeuverRecommendation> telemetryStructToManeuverRecommendation(const std::shared_ptr<Timebase_Telemetry_Waypoint>& car);
    static OptimizerEngine* getEngine();
    static double getHeadingDelta(double h0, double h1);
    static double safeHeadingValue(double heading);

    std::shared_ptr<std::thread> getThread();
    void startManeuverFeedback();
    void pauseManeuverFeedback();
    void updateSimulationState(std::unique_ptr<std::list<std::shared_ptr<RoadUser>>> cars);
    void removeFromSimulation(const std::string& uuid);
    std::list<std::shared_ptr<Timebase_Telemetry_Waypoint>> getSimulationResult();

};

#endif
