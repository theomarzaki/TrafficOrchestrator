//
// Created by jab on 27/05/19.
//

#include "include/optimizer_engine.h"

#include <mapper.h>
#include <network_interface.h>
#include <protocol.h>
#include <physx.h>
#include <stack>
#include <algorithm>

#define ROAD_DEFAULT_SPEED 20
#define MAX_ACCELERATION 2.0

#define HEADING_CONFIDENCE_AGAINST_ROAD_ANGLE 0.4
#define HUMAN_LATENCY_FACTOR 2000
#define LATENCY_DROP_FACTOR 2000
#define HEADING_REAJUST_UNIT 10.0
#define SPEED_REAJUST_UNIT 100.0
#define GPS_SIGNIFCAND 7
#define CAR_SIZE_REDUCTION_UNIT 10 // decimeter to meter

#define HIGHWAY_LANE_NUMBER 2
#define INSERTION_LANE_NUMBER 2

OptimizerEngine::OptimizerEngine() {
    kill.store(false);
    setBatch(INTERVAL_TIME);
}

void OptimizerEngine::killOptimizer() {
    kill.store(true);
}

void OptimizerEngine::startManeuverFeedback() {
    fence=true;
    cv.notify_all();
}

void OptimizerEngine::pauseManeuverFeedback() {
    fence=false;
    cv.notify_all();
}

std::shared_ptr<std::thread> OptimizerEngine::getThread() {
    return optimizerT;
}

void OptimizerEngine::setBatch(size_t interval) {
    optimizerT = std::make_shared<std::thread>([=]() mutable {
        while (!kill) {
            std::unique_lock<std::mutex> lock(pause);
            cv.wait(lock,[=]{return fence;});
            locker.lock();
            auto cars{this->getSimulationResult()};
            for (auto &car : cars) {
                logger::dumpToFile(Protocol::createRUDDescription(telemetryStructToManeuverRecommendation(*car)));
                NetworkInterface::sendTCP(Protocol::createManeuverJSON(telemetryStructToManeuverRecommendation(*car)));
            }
            logger::write("[INFOS] Maneuver send -> "+std::to_string(cars.size())+" cars reached\n");
            locker.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(interval));
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
            static_cast<double>(car->getHeading()/HEADING_REAJUST_UNIT),
            car->getSpeed()/SPEED_REAJUST_UNIT,
            static_cast<double>(car->getAcceleration()),
            static_cast<double>(car->getYawRate()),
            ROAD_DEFAULT_SPEED,
            Physx::getCarMass(car->getHeight()/CAR_SIZE_REDUCTION_UNIT,car->getLength()/CAR_SIZE_REDUCTION_UNIT,car->getWidth()/CAR_SIZE_REDUCTION_UNIT)
    };
}

double OptimizerEngine::mergeHeading(double h0, double h1) {
    if (h0 < h1) {
        auto buff{h0};
        h0 = safeHeadingValue(h1);
        h1 = safeHeadingValue(buff);
    }
    auto delta{getHeadingDelta(h0,h1)};
    auto adjust{delta/2};
    if (std::fabs(h0 - h1) > 180) {
        return h1 - adjust > 0 ? h1 - adjust : h0 + adjust;
    } else {
        return h0 - adjust;
    }
}

double OptimizerEngine::getHeadingDelta(double h0, double h1) {
    if (h0 < h1) {
        auto buff{h0};
        h0 = h1;
        h1 = buff;
    }
    auto delta{std::fabs(h0 - h1)};
    return delta < 180.0 ? delta : 360 - delta;
}

double OptimizerEngine::safeHeadingValue(double heading) {
    auto value{std::fmod(heading,360)};
    return value < 0 ? 360+value : value;
}

void OptimizerEngine::updateSimulationState(std::unique_ptr<std::list<std::shared_ptr<RoadUser>>> cars) {
    for (auto& car : *cars ) {
        auto buff{createTelemetryElementFromRoadUser(car)};
        if (game.find(car->getUuid()) != game.end()) {
            game.erase(car->getUuid());
        }
        game.insert({car->getUuid(),buff});
    }
}

void OptimizerEngine::removeFromSimulation(const std::string& uuid) {
    game.erase(uuid);
}

std::shared_ptr<ManeuverRecommendation> OptimizerEngine::telemetryStructToManeuverRecommendation(const Timebase_Telemetry_Waypoint& car) {
    auto mergingManeuver{std::make_shared<ManeuverRecommendation>()};
    auto speed{static_cast<uint16_t>(car.speed * SPEED_REAJUST_UNIT)};
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
                                  std::string(std::to_string(mergingManeuver->getTimestamp())));

    auto waypoint{std::make_shared<Waypoint>()};
    waypoint->setTimestamp(car.timestamp);
    waypoint->setLongitude(longitude);
    waypoint->setLatitude(latitude);
    waypoint->setSpeed(speed);
    waypoint->setLanePosition(car.laneId);
    waypoint->setHeading(static_cast<uint16_t>(car.heading*HEADING_REAJUST_UNIT));
    mergingManeuver->addWaypoint(waypoint);

    return mergingManeuver;
}

Timebase_Telemetry_Waypoint OptimizerEngine::getPositionOnRoadInInterval(Timebase_Telemetry_Waypoint car, int64_t interval, int64_t timenow) {
    int64_t deltaTime = timenow - car.timestamp;//time - car.second.timestamp

    double distance{Mapper::getDistance(car.speed,car.accelleration, (deltaTime + interval)/1000.0)};

    auto descriptor = Mapper::getMapper()->getPositionDescriptor(car.coordinates.latitude,car.coordinates.longitude,1);

    car.laneId = descriptor->laneId;

    auto coord{Mapper::getMapper()->followTheLaneWithDistance(car.coordinates,distance,1)};

    car.timestamp = timenow + HUMAN_LATENCY_FACTOR;
    car.heading = coord.heading;
    car.coordinates = {
            coord.latitude,
            coord.longitude
    };

    return car;
}

Timebase_Telemetry_Waypoint OptimizerEngine::forceCarMerging(Timebase_Telemetry_Waypoint car, int64_t interval, int64_t timenow) {
    int64_t deltaTime = timenow - car.timestamp;

    double distance;
    double deltaT{(deltaTime + interval)/1000.0};

    std::cout << "L " << car.speed << std::endl;

    double deltaV{car.max_speed - car.speed};
    double needTime{(deltaV / MAX_ACCELERATION)};
    if (car.speed < car.max_speed) { // TODO Need to go in Physx
        if (needTime < deltaT) {
            distance = (deltaT-needTime)*car.max_speed+(deltaT-(deltaT-needTime))*car.speed+0.5*MAX_ACCELERATION*needTime*needTime;
            car.speed = car.max_speed;
        } else {
            auto speed{MAX_ACCELERATION*deltaT+car.speed};
            distance = (speed + 0.5*MAX_ACCELERATION*deltaT)*deltaT;
            car.speed = speed;
        }
    } else {
        needTime *= -1;
        if (needTime < deltaT) {
            distance = deltaT*car.max_speed+0.5*MAX_ACCELERATION*needTime*needTime;
            car.speed = car.max_speed;
        } else {
            auto speed{car.speed-MAX_ACCELERATION*deltaT};
            distance = (speed + 0.5*MAX_ACCELERATION*deltaT)*deltaT;
            car.speed = speed;
        }
    }

    car.accelleration = 0;

    std::shared_ptr<Gps_View> gps{Mapper::getMapper()->getCoordinatesBydistanceAndRoadPath(car.coordinates.latitude,car.coordinates.longitude,distance,car.heading,15.0)};

    car.timestamp = timenow + HUMAN_LATENCY_FACTOR;
    car.coordinates.latitude = gps->latitude;
    car.coordinates.longitude = gps->longitude;
    car.heading = gps->heading;

    return car;
}

std::list<std::shared_ptr<Timebase_Telemetry_Waypoint>> OptimizerEngine::getSimulationResult() {
    std::map<std::string,std::shared_ptr<Timebase_Telemetry_Waypoint>> carStack;
    std::vector<std::shared_ptr<Graph_Element>> graphList;
    auto recos{std::list<std::shared_ptr<Timebase_Telemetry_Waypoint>>()};
    std::list<std::string> erase;

    std::string merginCarUuid;

    for (auto& car: game) {
        int64_t time{std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()};
        if ((time - car.second.timestamp) < LATENCY_DROP_FACTOR) {
            if (car.second.laneId == 0 and Mapper::getMapper()->getPositionDescriptor(car.second.coordinates.latitude,car.second.coordinates.longitude)->laneId == car.second.laneId) {
                car.second = forceCarMerging(car.second,HUMAN_LATENCY_FACTOR,time);
                merginCarUuid = car.second.uuid;
            } else {
                car.second = getPositionOnRoadInInterval(car.second,HUMAN_LATENCY_FACTOR,time);
            }
            carStack.insert({car.second.uuid,std::make_shared<Timebase_Telemetry_Waypoint>(car.second)});
        } else {
            erase.push_back(car.first);
        }
    }

    std::vector<std::shared_ptr<Timebase_Telemetry_Waypoint>> sortedCars;

    while (!carStack.empty()) {
        auto sickSheep{carStack.begin()->second};
        for (auto& element: carStack) {
            if (Mapper::getMapper()->isItBehindAGpsOnSameRoadPath(element.second->coordinates,sickSheep->coordinates)) {
                sickSheep = element.second;
            }
        }
        carStack.erase(sickSheep->uuid);
        sortedCars.push_back(sickSheep);
    }

    if (!sortedCars.empty()) {
        auto numberOfLanes{HIGHWAY_LANE_NUMBER+INSERTION_LANE_NUMBER};

        std::vector<int> lastLanesElement(numberOfLanes);
        for (long i=0; i<numberOfLanes; i++) { // Find first element of each lanes
            for (long x=sortedCars.size()-1; x >= 0; x--) {
                if (sortedCars.at(x)->laneId == i) {
                    lastLanesElement.push_back(x);
                }
            }
        }

        for (long i=sortedCars.size()-1; i >= 0; i--) {
            graphList.push_back(std::make_shared<Graph_Element>());
        }

        std::shared_ptr<Graph_Element> graphHead;
        auto head{graphList.at(0)};

        for (long i=sortedCars.size()-1, x=0; i >= 0; i--,x++) {
            auto car {sortedCars.at(i)};

            head = graphList.at(x);
            if (car->uuid == merginCarUuid) {
                graphHead = head;
            }
            head->telemetry = car;

            if (i == static_cast<long>(sortedCars.size()-1)) {
                for (const auto& elem: lastLanesElement) {
                    head->in_front_neighbours.push_back(graphList.at(elem));
                }
            } else if (i == 0) {
                for (const auto& elem: lastLanesElement) {
                    head->behind_neighbours.push_back(graphList.at(elem));
                }
            } else {
                for (const auto& elem: lastLanesElement) {
                    head->behind_neighbours.push_back(graphList.at(elem));
                }
                for (int z=0; z<numberOfLanes; z++) {
                    for (unsigned long y=i; y > 0; y--) {
                        if (sortedCars.at(y)->laneId == z) {
                            head->in_front_neighbours.push_back(graphList.at(y));
                        }
                    }
                }
            }
            lastLanesElement.at(car->laneId) = i;
        }
    }

//    head = graphHead;
//    std::vector<std::shared_ptr<Graph_Element>> mutableGraphList(graphList);
//    while(!mutableGraphList.empty()) { // Where the magic happen, not so Magic tho'.
//        if (head->telemetry->connected) {
//            if (head->telemetry->laneId == 0) {
//
//            } else {
//
//            }
//        }
//    }

    //TODO Optimise graph only with connected

    for (auto& car: graphList) {
        if (car->telemetry->connected) {
            std::cout << std::setprecision(GPS_SIGNIFCAND) << std::fixed << car->telemetry->uuid << " " << car->telemetry->laneId << " " << car->telemetry->coordinates.latitude << "," << car->telemetry->coordinates.longitude << "\n";
            recos.push_back(car->telemetry);
        }
    }

    std::cout << std::endl;

    for (auto& uuid : erase) game.erase(uuid);
    return recos;
}
