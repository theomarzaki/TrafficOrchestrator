//
// Created by jab on 27/05/19.
//

#include "include/optimizer_engine.h"

#include <mapper.h>
#include <network_interface.h>

#define HEADING_CONFIDENCE_AGAINST_ROAD_ANGLE 0.6
#define HUMAN_LATENCY_FACTOR 2000
#define LATENCY_DROP_FACTOR 800
#define HEADING_REAJUST_RATIO 10.0
#define SPEED_REAJUST_RATIO 100.0
#define GPS_SIGNIFCAND 7

OptimizerEngine::OptimizerEngine() {
    kill.store(false);
    pause.store(true);
    setBatch(INTERVAL_TIME);
}

void OptimizerEngine::killOptimizer() {
    kill.store(true);
}

void OptimizerEngine::startManeuverFeedback() {
    pause.store(false);
}

void OptimizerEngine::pauseManeuverFeedback() {
    pause.store(true);
}

std::shared_ptr<std::thread> OptimizerEngine::getThread() {
    return optimizerT;
}

void OptimizerEngine::setBatch(size_t interval) {
    optimizerT = std::make_shared<std::thread>([=]() mutable {
        while (!kill) {
            while (!pause) {
                locker.lock();
                auto cars{this->getSimulationResult()}; // TODO Server send
                for (auto &car : *cars) {
                    SendInterface::sendTCP(SendInterface::createManeuverJSON(telemetryStructToManeuverRecommendation(car)));
                }
                if (!cars->empty()) logger::write("[INFOS] Maneuver send -> "+std::to_string(cars->size())+" cars reached");
                locker.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(interval));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
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
                    (car->getLatitude() / std::pow(10,GPS_SIGNIFCAND)),
                    (car->getLongitude() / std::pow(10,GPS_SIGNIFCAND)),
            },
            car->getUuid(),
            car->getConnected(),
            static_cast<int>(car->getLanePosition()),
            static_cast<int64_t >(car->getTimestamp()),
            static_cast<double>(car->getHeading()/HEADING_REAJUST_RATIO),
            car->getSpeed()/SPEED_REAJUST_RATIO,
            static_cast<double>(car->getAcceleration()),
            static_cast<double>(car->getYawRate()),
    };
}

double OptimizerEngine::mergeHeading(double h0, double h1) {
    h0 *= HEADING_CONFIDENCE_AGAINST_ROAD_ANGLE;
    h1 += h1*(1-HEADING_CONFIDENCE_AGAINST_ROAD_ANGLE);
    if ( std::fabs(h0-h1) > 180 ) {
        return std::fabs(h0-h1)/2;
    } else if (h0 > h1 ) {
        return std::fmod(h0+(-(h0-360)+h1)/2,360);
    } else {
        return std::fmod(h1+(-(h1-360)+h0)/2,360);
    }
}

void OptimizerEngine::updateSimulationState(std::unique_ptr<std::list<std::shared_ptr<RoadUser>>> cars) {
    for (auto& car : *cars ) {
        if (game.find(car->getUuid()) == game.end()) {
            game.insert({car->getUuid(),createTelemetryElementFromRoadUser(car)});
        } else {
            auto buff{createTelemetryElementFromRoadUser(car)};
            game[car->getUuid()] = buff; //TODO Filter removed
        }
    }
}

void OptimizerEngine::removeFromSimulation(const std::string& uuid) {
    game.erase(uuid);
}

std::shared_ptr<ManeuverRecommendation> OptimizerEngine::telemetryStructToManeuverRecommendation(const Timebase_Telemetry_Waypoint& car) {
    auto mergingManeuver{std::make_shared<ManeuverRecommendation>()};
    auto speed{static_cast<uint16_t>(car.speed * SPEED_REAJUST_RATIO)};
    auto latitude{static_cast<int32_t>(car.coordinates.latitude * std::pow(10,GPS_SIGNIFCAND))};
    auto longitude{static_cast<int32_t>(car.coordinates.longitude * std::pow(10,GPS_SIGNIFCAND))};

    mergingManeuver->setTimestamp(car.timestamp);
    mergingManeuver->setUuidVehicle(car.uuid);
    mergingManeuver->setUuidTo(car.uuid);
    mergingManeuver->setTimestampAction(car.timestamp);
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
    waypoint->setHeading(static_cast<uint16_t>(car.heading*HEADING_REAJUST_RATIO));
    mergingManeuver->addWaypoint(waypoint);

    return mergingManeuver;
}

std::unique_ptr<std::list<Timebase_Telemetry_Waypoint>> OptimizerEngine::getSimulationResult() {
    auto cars{std::make_unique<std::list<Timebase_Telemetry_Waypoint>>()};
    std::list<std::string> erase;
    for(auto& car: game) {
        int64_t time{std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()};
        if ((time - car.second.timestamp) < LATENCY_DROP_FACTOR) {
            if(car.second.connected) { //TODO MAGIC

                int64_t deltaTime = time - car.second.timestamp;//time - car.second.timestamp;

                Gps_Point gps {
                    car.second.coordinates.latitude,
                    car.second.coordinates.longitude,
                };

                double distance{Mapper::getDistance(car.second.speed,car.second.accelleration, (deltaTime + HUMAN_LATENCY_FACTOR)/1000.0)};
                std::cout << "DANK "<< distance << " " << car.second.speed <<  "\n";

                auto descriptor{Mapper::getMapper()->getPositionDescriptor(car.second.coordinates.latitude,car.second.coordinates.longitude)};

                car.second.timestamp = time + HUMAN_LATENCY_FACTOR;

                car.second.coordinates = Mapper::projectGpsPoint(gps, distance, mergeHeading(car.second.heading,descriptor.heading));

                cars->push_back(car.second);
            }
        } else {
            erase.push_back(car.first);
        }
    }
    for (auto& uuid : erase ) game.erase(uuid);
    return std::move(cars);
}
