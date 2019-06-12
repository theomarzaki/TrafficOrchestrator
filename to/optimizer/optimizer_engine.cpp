//
// Created by jab on 27/05/19.
//

#include "include/optimizer_engine.h"

#include <mapper.h>

#define SPEED_REAJUST_RATIO 100
#define SIGNIFCAND 7

OptimizerEngine::OptimizerEngine() {
    kill.store(true);
    launchBatch(kill, INTERVAL_TIME);
}

void OptimizerEngine::stopManeuverFeedback(bool stop) {
    kill.store(!stop);
}

void OptimizerEngine::launchBatch(std::atomic_bool& kill, size_t interval) {
    std::async(std::launch::async,[=, &kill]() mutable {
        while (kill) {
            std::cout << "Calculate Maneuver feedback" << std::endl;
            auto cars{this->getSimulationResult()}; // TODO Server send
            auto recos{vector<std::shared_ptr<ManeuverRecommendation>>()};
            for(auto& car : *cars) {
                recos.push_back(telemetryStructToManeuverRecommendation(car));
            }
//            sendDataTCP();
            std::this_thread::sleep_for(std::chrono::milliseconds(interval));
        }
    });
}

std::shared_ptr<OptimizerEngine> OptimizerEngine::getEngine(){
    try {
        if (engine == nullptr) {
            engine = std::shared_ptr<OptimizerEngine>(new OptimizerEngine()); // private visibility fix
        }
        return engine;
    } catch (const std::exception& e) {
        logger::write("[ERROR] OptimizerEngine initializer");
    }
    return nullptr;
}

Timebase_Telemetry_Waypoint OptimizerEngine::createTelemetryElementFromRoadUser(const std::shared_ptr<RoadUser>& car) {
    return {
            Gps_Point {
                    (car->getLatitude() / std::pow(10,SIGNIFCAND)),
                    (car->getLongitude() / std::pow(10,SIGNIFCAND)),
            },
            car->getUuid(),
            car->getConnected(),
            static_cast<int>(car->getLanePosition()),
            static_cast<int>(car->getTimestamp()),
            static_cast<double>(car->getHeading()),
            static_cast<double>(car->getSpeed()),
            static_cast<double>(car->getAcceleration()),
            static_cast<double>(car->getYawRate()),
    };
}

std::shared_ptr<ManeuverRecommendation> telemetryStructToManeuverRecommendation(Timebase_Telemetry_Waypoint car) {
    auto mergingManeuver{std::make_shared<ManeuverRecommendation>()};
    auto timestamp{std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()};

    auto speed{static_cast<uint4>(car.speed * SPEED_REAJUST_RATIO)};
    auto latitude{static_cast<uint32_t>(car.coordinates.latitude * std::pow(10,SIGNIFCAND))};
    auto longitude{static_cast<uint32_t>(car.coordinates.longitude * std::pow(10,SIGNIFCAND))};

    mergingManeuver->setTimestamp(timestamp);
    mergingManeuver->setUuidVehicle(car.uuid);
    mergingManeuver->setUuidTo(car.uuid);
    mergingManeuver->setTimestampAction(timestamp);
    mergingManeuver->setLongitudeAction(longitude);
    mergingManeuver->setLatitudeAction(latitude);
    mergingManeuver->setSpeedAction(speed);
    mergingManeuver->setLanePositionAction(car.laneId);
    mergingManeuver->setMessageID(std::string(mergingManeuver->getOrigin()) + "/" + std::string(mergingManeuver->getUuidManeuver()) + "/" +
                                  std::string(to_string(mergingManeuver->getTimestamp())));

    auto waypoint{std::make_shared<Waypoint>()};
    waypoint->setTimestamp(car.timestamp);
    waypoint->setLongitude(longitude);
    waypoint->setLatitude(latitude);
    waypoint->setSpeed(speed);
    waypoint->setLanePosition(car.laneId);
    waypoint->setHeading(static_cast<uint16_t>(car.heading));
    mergingManeuver->addWaypoint(waypoint);

    return mergingManeuver;
}

void OptimizerEngine::updateSimulationState(std::unique_ptr<std::list<std::shared_ptr<RoadUser>>> cars) {
    for (auto& car : *cars ) {
        if (game.find(car->getUuid()) == game.end()) {
            game.insert({car->getUuid(),createTelemetryElementFromRoadUser(car)});
        } else {
            auto buff{createTelemetryElementFromRoadUser(car)};
            if (buff.timestamp > game[car->getUuid()].timestamp) {
                game[car->getUuid()] = buff;
            }
        }
    }
}

std::unique_ptr<std::list<Timebase_Telemetry_Waypoint>> OptimizerEngine::getSimulationResult() {
    auto cars{std::make_unique<std::list<Timebase_Telemetry_Waypoint>>()};
    for(auto& car: game) {
        if(car.second.connected) { //TODO MAGIC
            cars->push_back(car.second);
        }
    }
    return std::move(cars);
}
