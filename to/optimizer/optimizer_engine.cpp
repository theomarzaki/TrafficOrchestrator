//
// Created by jab on 27/05/19.
//

#include "include/optimizer_engine.h"

#include <mapper.h>
#include <network_interface.h>

#define HEADING_CONFIDENCE_AGAINST_ROAD_ANGLE 0.6
#define HUMAN_LATENCY_FACTOR 2500
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
                for (auto &car : cars) {
                    SendInterface::sendTCP(SendInterface::createManeuverJSON(telemetryStructToManeuverRecommendation(*car)));
                }
                if (!cars.empty()) logger::write("[INFOS] Maneuver send -> "+std::to_string(cars.size())+" cars reached");
                locker.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(interval));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        }
    });
}

OptimizerEngine* OptimizerEngine::getEngine(){
    try {
        if (engine == nullptr) {
            engine = new OptimizerEngine(); // private visibility fix
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

Timebase_Telemetry_Waypoint OptimizerEngine::getPositionOnRoadInInterval(Timebase_Telemetry_Waypoint car, int64_t interval, int64_t timenow) {
    int64_t deltaTime = timenow - car.timestamp;//time - car.second.timestamp;4
    Gps_Point gps {
            car.coordinates.latitude,
            car.coordinates.longitude,
    };

    double distance{Mapper::getDistance(car.speed,car.accelleration, (deltaTime + interval)/1000.0)};
    std::cout << "DANK "<< distance << " " << car.speed <<  "\n";

    Mapper::Gps_Descriptor descriptor = Mapper::getMapper()->getPositionDescriptor(car.coordinates.latitude,car.coordinates.longitude);

    car.timestamp = timenow + HUMAN_LATENCY_FACTOR;
    car.coordinates = Mapper::projectGpsPoint(gps, distance, car.heading);

    return car;
}

std::list<std::shared_ptr<Timebase_Telemetry_Waypoint>> OptimizerEngine::getSimulationResult() {
    auto cars{std::list<std::shared_ptr<Timebase_Telemetry_Waypoint>>()};
    auto recos{std::list<std::shared_ptr<Timebase_Telemetry_Waypoint>>()};
    std::list<std::string> erase;
    for(auto& car: game) {
        int64_t time{std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()};
        if ((time - car.second.timestamp) < LATENCY_DROP_FACTOR) {
            car.second = getPositionOnRoadInInterval(car.second,HUMAN_LATENCY_FACTOR,time);
            if(car.second.connected) {
                cars.push_back(std::make_shared<Timebase_Telemetry_Waypoint>(car.second));
            }
        } else {
            erase.push_back(car.first);
        }
    }

    //TODO MAGIC

    for(auto& car: cars) {
        if(car->connected) {
            recos.push_back(car);
        }
    }
    for (auto& uuid : erase ) game.erase(uuid);
    return recos;
}
