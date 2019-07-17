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

#include <network_interface.h>
#include <maneuver_recommendation.h>
#include <road_user.h>
#include <mapper.h>
#include <gpstools.h>


class OptimizerEngine {
private:
    inline static OptimizerEngine* engine;

    std::map<std::string,Timebase_Telemetry_Waypoint> game;
    std::atomic_bool kill;
    std::shared_ptr<std::thread> optimizerT;
    std::condition_variable cv;

    std::mutex pause;

    OptimizerEngine();

    void setBatch(size_t interval);

    struct Stashed_Telemetry {
        bool connected;
        double theta;
        int laneId;
        double heading;
        double distante_to_parent;
        double speed;
    };

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
    static std::shared_ptr<ManeuverRecommendation> telemetryStructToManeuverRecommendation(const Timebase_Telemetry_Waypoint& car);
    static OptimizerEngine* getEngine();
    static double mergeHeading(double h0, double h1);
    static double getHeadingDelta(double h0, double h1);
    static double safeHeadingValue(double heading)

    std::shared_ptr<std::thread> getThread();
    void killOptimizer();
    void startManeuverFeedback();
    void pauseManeuverFeedback();
    void updateSimulationState(std::unique_ptr<std::list<std::shared_ptr<RoadUser>>> cars);
    void removeFromSimulation(const std::string& uuid);
    std::list<std::shared_ptr<Timebase_Telemetry_Waypoint>> getSimulationResult();

};

#endif
