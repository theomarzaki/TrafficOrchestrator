//
// Created by Johan Maurel <johan.maurel@orange.com> on 15/04/19.
// Copyright (c) Orange Labs all rights reserved.
//

#include "include/mapper.h"

Mapper::Mapper() {
    pathToMap = pathToMap.empty() ? MONTLHERY_MAP : pathToMap; // By default

    std::string line, jsonString;
    std::ifstream file("./include/"+pathToMap+".json");
    if (file.is_open()) {
        while( getline(file,line)) {
            jsonString += line + '\n';
        }
        file.close();
    }

    auto json = std::make_unique<rapidjson::Document>();
    if (!jsonString.empty()) {
        json->Parse(jsonString.c_str());
        createMapContainer(std::move(json));
    } else {
        throw std::runtime_error("[ERROR] File is empty");
    }
}

void Mapper::createMapContainer(std::unique_ptr<rapidjson::Document> json) { // TODO Change array topography
    if (json->IsObject()) {
        numberOfRoad = json->HasMember("numberOfRoad") ? json->FindMember("numberOfRoad")->value.GetInt() : 0;
        for (auto& road : json->FindMember("roads")->value.GetArray()) {
            Road_Descriptor roadStruct;
            roadStruct.id = road.HasMember("_id") ? road["_id"].GetInt() : -1;
            roadStruct.numberOfLanes = road.HasMember("numberOfLanes") ? road["numberOfLanes"].GetInt() : 0;
            roadStruct.name = road["name"].GetString();
            roadStruct.speed = road.HasMember("speed") ? road["speed"].GetInt() : 80;
            roadStruct.type = road.HasMember("type") and std::string(road["type"].GetString()) == "loop" ? Road_Type::LOOP : Road_Type::STRIPE;

            for (auto& lane : road["lanes"].GetArray()) {
                Lane_Descriptor laneStruct;
                laneStruct.id = lane.HasMember("_id") ? lane["_id"].GetInt() : -1;
                laneStruct.size = lane.HasMember("size") ? lane["size"].GetFloat() : 0;

                for (auto& node : lane["points"].GetArray()) {
                    Lane_Node nodeStruct;
                    nodeStruct.latitude = node.HasMember("lat") ? node["lat"].GetDouble() : 120.0; // Aka impossible
                    nodeStruct.longitude = node.HasMember("long") ? node["long"].GetDouble() : 220.0;
                    laneStruct.nodes.push_back(std::make_shared<Lane_Node>(nodeStruct));
                }
                roadStruct.lanes.insert({laneStruct.id,laneStruct});
            }
            m_roads.push_back(roadStruct);
        }
    } else {
        throw std::runtime_error("[ERROR] File is not a valid JSON");
    }
}

bool Mapper::setMap(const std::string& mapPath) {
    if ( access( std::string("./include/"+mapPath+".json").c_str(), F_OK ) > -1) {
        std::cout << "[INFO] Map path set" << std::endl;
        pathToMap = mapPath;
        return true;
    } else {
        std::cout << "[ERROR] '" << mapPath.c_str() << "' don't exist in the include folder" << std::endl;
        return false;
    }
}

std::shared_ptr<Mapper> Mapper::getMapper(){
    try {
        if (mapper == nullptr) {
            mapper = std::shared_ptr<Mapper>(new Mapper()); // private visibility fix
        }
        return mapper;
    } catch (const std::exception& e) {
        std::cout << "[ERROR] Mapper initializer : " << e.what() << std::endl;
    }
    return nullptr;
}

double Mapper::toRadian(double value) {
    return value * M_PI / 180.0;
}

double Mapper::toDegree(double value) {
    return value * 180.0 / M_PI;
}

double Mapper::distanceBetween2GPSCoordinates(double latitude, double longitude,
                                             double latitude2, double longitude2) {  // in meters not in imperials shits
    latitude = toRadian(latitude);
    longitude = toRadian(longitude);
    latitude2 = toRadian(latitude2);
    longitude2 = toRadian(longitude2);

    auto deltalat {latitude2 - latitude};
    auto deltalong {longitude2 - longitude};

    double a = sin(deltalat/2) * sin(deltalat/2) + cos(latitude) * cos(latitude2) * sin(deltalong/2) * sin(deltalong/2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));

    return EARTH_RADIUS * c;
}

double Mapper::getSpeed(  double speed,
                          double acceleration,
                          double time ) {
    return speed + acceleration*time;
}

double Mapper::getDistance(  double speed,
                            double acceleration,
                            double time) {
    return (speed + 0.5 * acceleration * time) * time;
}

Gps_Point Mapper::projectGpsPoint(  Gps_Point coord,
                                    double distance,
                                    double heading ) {
    heading = toRadian(heading);
    double lat{toRadian(coord.latitude)};
    double lon{toRadian(coord.longitude)};
    double gradian{distance / EARTH_RADIUS};

    // Latitude calculus before longitude
    double la{std::asin(std::sin(lat) * std::cos(gradian) + std::cos(lat) * std::sin(gradian) * std::cos(heading))};

    return {
            toDegree(la),
            toDegree( std::fmod( lon + std::atan2( std::sin(heading) * std::sin(gradian) * std::cos(lat), std::cos(gradian) - std::sin(lat) * std::sin(la)) + M_PI, 2 * M_PI) - M_PI),
    };
}

double Mapper::getHeading(   double xP, double yP,
                                double xH, double yH ) {
    auto degree{toDegree(std::atan2(yH - yP, xH - xP))};
    degree = degree < 0 ? 270 + degree : degree -90;
    return degree < 0 ? -degree : 360 - degree;
}

Point_2D Mapper::transformCoordinatesFromGPSTo2DGrid(double latitudeBase, double longitudeBase, double latitude, double longitude) {
    auto x{distanceBetween2GPSCoordinates(latitudeBase, longitudeBase, latitudeBase, longitude)};
    auto y{distanceBetween2GPSCoordinates(latitudeBase, longitudeBase, latitude, longitudeBase)};
    x = longitude < longitudeBase ? -x : x;
    y = latitude < latitudeBase ? -y : y;
    return {
        x,
        y,
    };
}

double Mapper::distanceBetweenAPointAndAStraightLine(   double xP, double yP,
                                                        double xL, double yL,
                                                        double xH, double yH ) {
    auto coef = (yH - yL) / (xH - xL);
    auto offset = coef*xL - yL;
    return std::fabs(- coef*xP + yP - offset) / std::sqrt( std::pow(coef,2) + 1 );
}

std::shared_ptr<Mapper::Gps_Descriptor> Mapper::getPositionDescriptor(double latitude, double longitude, int forcedRoadID, int forcedLaneID) {

    Gps_Descriptor nearestDescription;
    nearestDescription.roadId = -1;
    nearestDescription.laneId = -1;
    nearestDescription.nodeId = -1;
    nearestDescription.roadName = "";
    nearestDescription.state = Mapper_Result_State::ERROR;

    auto distance = std::numeric_limits<double>::max();
    int maxIndex = 0;

    auto listRoads = m_roads;
    try {
        if (forcedRoadID > -1) {
            auto buff{std::vector<Road_Descriptor>()};
            auto road{m_roads.at(forcedRoadID)};
            if (forcedLaneID > -1) {
                auto lanes{std::map<int,Lane_Descriptor>()};
                auto lane = road.lanes.find(forcedLaneID);
                if (lane != road.lanes.end()) {
                    lanes.insert({lane->first,lane->second});
                    road.lanes = lanes;
                } else {
                    logger::write(std::string("[ERROR] The lane "+std::to_string(forcedLaneID)+" don't exist in the map."));
                }
            }
            buff.push_back(road);
            listRoads = buff;
        }
    } catch(const std::exception& e) {
        if (typeid(e) == typeid(std::out_of_range)) {
            logger::write(std::string("[ERROR] Seems that the road id "+std::to_string(forcedRoadID)+" don't exist"));
        } else {
            logger::write("[ERROR] Unknown error in map listing process");
        }
    }

    for (auto& road : listRoads) { // TODO Change basic optimum search
        for (auto& lane : road.lanes) {
            for (unsigned long i=0; i < lane.second.nodes.size(); i++) { // TODO Find match with sector vectors
                double nodeDistance = distanceBetween2GPSCoordinates(lane.second.nodes.at(i)->latitude, lane.second.nodes.at(i)->longitude, latitude, longitude);
                if (nodeDistance < distance) {
                    distance = nodeDistance;
                    nearestDescription.roadId = road.id;
                    nearestDescription.laneId = lane.second.id;
                    nearestDescription.state = Mapper_Result_State::OUT_OF_ROAD;
                    nearestDescription.roadName = road.name;
                    nearestDescription.speed = road.speed;
                    maxIndex = size(lane.second.nodes) - 1;
                    nearestDescription.nodeId = i;
                }
            }
        }
    }

    if (nearestDescription.nodeId > -1) {
        if (distance < OUT_OF_MAP_VALUE) { // TODO Find the extremum of the serie and define the limit with it.
            auto lane{m_roads.at(nearestDescription.roadId).lanes.find(nearestDescription.laneId)->second};
            auto node{lane.nodes.at(nearestDescription.nodeId)};

            int buffIndex;

            Lane_Node compareNode;

            Lane_Node previousNode;
            Lane_Node nextNode;

            Lane_Node roadDirectionThirdNode;
            Lane_Node roadDirectionFirstNode;
            Lane_Node roadDirectionSecondNode;

            if (nearestDescription.nodeId == 0) {
                buffIndex = nearestDescription.nodeId +2;
                previousNode = *lane.nodes.at(maxIndex);
                nextNode = *lane.nodes.at(nearestDescription.nodeId +1);
            } else if (nearestDescription.nodeId == maxIndex) {
                buffIndex = 1;
                previousNode = *lane.nodes.at(nearestDescription.nodeId - 1);
                nextNode = *lane.nodes.at(0);
            } else if (nearestDescription.nodeId == maxIndex-1) {
                buffIndex = 0;
                previousNode = *lane.nodes.at(nearestDescription.nodeId -1);
                nextNode = *lane.nodes.at(nearestDescription.nodeId +1);
            } else {
                buffIndex = nearestDescription.nodeId+2;
                previousNode = *lane.nodes.at(nearestDescription.nodeId -1);
                nextNode = *lane.nodes.at(nearestDescription.nodeId +1);
            }

            auto distanceToPrevious = distanceBetween2GPSCoordinates(previousNode.latitude, previousNode.longitude, latitude, longitude);
            auto distanceToNext = distanceBetween2GPSCoordinates(nextNode.latitude, nextNode.longitude, latitude, longitude);

            if (distanceToPrevious < distanceToNext) {
                compareNode = previousNode;
                roadDirectionFirstNode = previousNode;
                roadDirectionSecondNode = *node;
                roadDirectionThirdNode = nextNode;
            } else {
                compareNode = nextNode;
                roadDirectionThirdNode = *lane.nodes.at(buffIndex);
                roadDirectionFirstNode = *node;
                roadDirectionSecondNode = nextNode;
            }

            auto xH{distanceBetween2GPSCoordinates(compareNode.latitude, compareNode.longitude, compareNode.latitude, node->longitude)};
            auto yH{distanceBetween2GPSCoordinates(compareNode.latitude, compareNode.longitude, node->latitude, compareNode.longitude)};
            auto xP{distanceBetween2GPSCoordinates(compareNode.latitude, compareNode.longitude, compareNode.latitude, longitude)};
            auto yP{distanceBetween2GPSCoordinates(compareNode.latitude, compareNode.longitude, latitude, compareNode.longitude)};

            xH = node->latitude < compareNode.latitude ? -xH : xH;
            yH = node->longitude < compareNode.longitude ? -yH : yH;

            xP = latitude < compareNode.latitude ? -xP : xP;
            yP = longitude < compareNode.longitude ? -yP : yP;

            auto nextNodeTransform{transformCoordinatesFromGPSTo2DGrid(roadDirectionFirstNode.latitude,roadDirectionFirstNode.longitude,roadDirectionSecondNode.latitude,roadDirectionSecondNode.longitude)};
            auto nextHeadingNodeTransform{transformCoordinatesFromGPSTo2DGrid(roadDirectionFirstNode.latitude,roadDirectionFirstNode.longitude,roadDirectionThirdNode.latitude,roadDirectionThirdNode.longitude)};

            nearestDescription.heading = getHeading(0, 0, nextNodeTransform.x, nextNodeTransform.y);
            nearestDescription.next_heading = getHeading(0, 0, nextHeadingNodeTransform.x, nextHeadingNodeTransform.y);

            auto distanceToMiddle{distanceBetweenAPointAndAStraightLine(xP, yP, 0, 0, xH, yH)};
            nearestDescription.distance_to_middle = distanceToMiddle;

            if (distanceToMiddle <= lane.size/2) { // TODO implement Square B-Spline distance check
                nearestDescription.state = Mapper_Result_State::OK;
            }
        } else {
            nearestDescription.state = Mapper_Result_State::OUT_OF_MAP;
            logger::write("[WARN] Coordinates out of map, at least "+std::to_string(OUT_OF_MAP_VALUE)+" m away => ("+std::to_string(latitude)+","+std::to_string(longitude)+")");
        }
    }

    return std::make_shared<Gps_Descriptor>(nearestDescription);
}

std::shared_ptr<Gps_View> Mapper::getCoordinatesBydistanceAndRoadPath(double latitude, double longitude, double distance, double heading, double maxHeading) {
    auto gps{getPositionDescriptor(latitude,longitude)};
    auto distanceToMiddle{getPositionDescriptor(latitude,longitude,1)->distance_to_middle};
    auto lane{m_roads.at(gps->roadId).lanes.find(gps->laneId)->second};
    auto node{lane.nodes.at(gps->nodeId)};
    auto nextNode{lane.nodes.at(gps->nodeId+1)};

     std::cout << distance << " " << distanceToMiddle << std::endl;

    if (distanceToMiddle > distance/2) {
        for (unsigned long i = 2; i < lane.nodes.size()-2; i++) {
            auto nextDistance{distanceBetween2GPSCoordinates(latitude, longitude, nextNode->latitude, nextNode->longitude)};
            if (nextDistance < distance) {
                distance -= nextDistance;
                latitude = nextNode->latitude;
                longitude = nextNode->longitude;
                nextNode = lane.nodes.at(gps->nodeId + i);
            } else {
                auto transform{transformCoordinatesFromGPSTo2DGrid(latitude, longitude, nextNode->latitude, nextNode->longitude)};
                auto guideline{getHeading(0, 0, transform.x, transform.y)};
                auto coord{projectGpsPoint({latitude, longitude}, distance, guideline)};
                Gps_View buff;
                buff.latitude = coord.latitude;
                buff.longitude = coord.longitude;
                buff.heading = guideline;
                return std::make_shared<Gps_View>(buff);
            }
        }
    } else {
        auto coord{findCrossingPointBetweenLaneAndGpsVector(1,1,{latitude,longitude}, heading, distance, maxHeading)};
        Gps_View buff;
        buff.latitude = coord->latitude;
        buff.longitude = coord->longitude;
        buff.heading = coord->heading;
        return std::make_shared<Gps_View>(buff);
    }
    return std::make_shared<Gps_View>();
}

Gps_Point Mapper::getGpsPointWith2DPointAndBaseGpsPoint(Gps_Point baseGps, double x, double y) {
    auto heading{getHeading(0,0,x,y)};
    auto distance{distanceBetweenAPointAndAStraightLine(0, 0, 0, 0, x, y)};
    return projectGpsPoint(baseGps,distance,heading);
}

std::optional<bool> Mapper::isItBehindAGpsOnSameRoadPath(Gps_Point carBase, Gps_Point carCompared) {
    auto gpsBase{getPositionDescriptor(carBase.latitude,carBase.longitude)};
    auto gpsCompared{getPositionDescriptor(carCompared.latitude,carCompared.longitude)};

    if (gpsBase->roadId == gpsCompared->roadId ) {
        if (gpsBase->nodeId < gpsCompared->nodeId) {
            return true;
        } else if (gpsBase->nodeId == gpsCompared->nodeId) {
            auto lanes{m_roads.at(gpsBase->roadId).lanes};
            auto laneBase{lanes.find(gpsBase->laneId)};
            auto laneCompared{lanes.find(gpsCompared->laneId)};

            int max = laneBase->second.nodes.size() - 1;
            int nextIndex{gpsBase->nodeId};

            if (gpsBase->nodeId == max) {
                nextIndex=0;
            } else if (gpsBase->nodeId == 0) {
                nextIndex=1;
            } else {
                nextIndex++;
            }

            auto nextNodeBase{laneBase->second.nodes.at(nextIndex)};
            auto nextNodeCompared{laneCompared->second.nodes.at(nextIndex)};

            return distanceBetween2GPSCoordinates(carBase.latitude, carBase.longitude, nextNodeBase->latitude, nextNodeBase->longitude) >
                   distanceBetween2GPSCoordinates(carCompared.latitude, carCompared.longitude, nextNodeCompared->latitude, nextNodeCompared->longitude);

        } else {
            return false;
        }
    } else {
        return std::nullopt;
    }
}

std::shared_ptr<Gps_View> Mapper::findCrossingPointBetweenLaneAndGpsVector(int roadId, int laneId, Gps_Point car, double heading, double maxDistance, double maxAngle) {
    auto gps{getPositionDescriptor(car.latitude,car.longitude,roadId,laneId)};
    auto lane{m_roads.at(gps->roadId).lanes.find(gps->laneId)->second};
    auto node{lane.nodes.at(gps->nodeId)};
    auto nextNode{lane.nodes.at(gps->nodeId+1)};
    auto pathFound{false};
    Gps_View gpsSolution;

    auto baseRoad{transformCoordinatesFromGPSTo2DGrid(car.latitude, car.longitude, node->latitude, node->longitude)};
    auto distantRoad{transformCoordinatesFromGPSTo2DGrid(car.latitude, car.longitude, nextNode->latitude, nextNode->longitude)};

    for (double angle=heading-maxAngle; angle < heading+maxAngle; angle+=1.0) { // TODO Can break
        auto mergingCarGps{projectGpsPoint({car.latitude,car.longitude},maxDistance,angle)};
        auto distantMergingPoints{transformCoordinatesFromGPSTo2DGrid(car.latitude, car.longitude, mergingCarGps.latitude, mergingCarGps.longitude)};

        double Ua{((distantRoad.x-baseRoad.x)*(-baseRoad.y)-(distantRoad.y-baseRoad.y)*(-baseRoad.x))/((distantRoad.y-baseRoad.y)*(distantMergingPoints.x)-(distantRoad.x-baseRoad.x)*(distantMergingPoints.y))};

        double Ub{((distantMergingPoints.x)*(-baseRoad.y)-(distantMergingPoints.y)*(-baseRoad.x))/((distantRoad.y-baseRoad.y)*(distantMergingPoints.x)-(distantRoad.x-baseRoad.x)*(distantMergingPoints.y))};

        if (Ua > 0.0 and Ua < 1.0 and Ub > 0.0 and Ub < 1.) {
            pathFound=true;
            auto Sx{Ua*distantMergingPoints.x};
            auto Sy{Ua*distantMergingPoints.y};
            auto coord{getGpsPointWith2DPointAndBaseGpsPoint(car,Sx,Sy)};
            gpsSolution = {
                coord.latitude,
                coord.longitude,
                getHeading(baseRoad.x,baseRoad.y,distantRoad.x,distantRoad.y)
            };
        }
    }
    if (pathFound) {
        return std::make_shared<Gps_View>(gpsSolution);
    } else {
        auto coord{projectGpsPoint({car.latitude, car.longitude}, maxDistance, gps->heading)};
        Gps_View buff {
            coord.latitude,
            coord.longitude,
            gps->heading
        };
        return std::make_shared<Gps_View>(buff); // Continue on the road
    }
}

int Mapper::numberOfLaneInCurrentGpsPoint(Gps_Point car) {
    return m_roads.at(getPositionDescriptor(car.latitude,car.longitude)->roadId).numberOfLanes;
}

std::optional<Mapper::Merging_Scenario> Mapper::getFakeCarMergingScenario(double latitude, double longitude, int laneId) {  // Beware that method is tweaked for our use case. Such as the the road = 1 and lane = 1.
    laneId = laneId != -1 ? 2 : 1; // Only two lanesId on the Highway (1 and 2).
    auto gps{getPositionDescriptor(latitude,longitude,1,laneId)}; // 1 = highway
    if (gps->state != Mapper_Result_State::OUT_OF_MAP) {
        auto nodes{m_roads.at(gps->roadId).lanes.find(laneId)->second.nodes}; // 1 = First lane
        long max = nodes.size()-1;
        long spread = nodes.size()/6; // size factor

        unsigned int indexFollowing = gps->nodeId - spread < 0 ? max + (gps->nodeId - spread + 1) : gps->nodeId - spread;
        unsigned int indexPreceeding = gps->nodeId + spread > max ? (gps->nodeId + spread) % max - 1 : gps->nodeId + spread;

        Gps_Point carPreceeding = {
                nodes.at(indexPreceeding)->latitude,
                nodes.at(indexPreceeding)->longitude,
        };

        Gps_Point carFollowing = {
                nodes.at(indexFollowing)->latitude,
                nodes.at(indexFollowing)->longitude,
        };

        Merging_Scenario ret = {
                carPreceeding,
                carFollowing,
        };

        return ret;
    } else {
        return std::nullopt;
    }
}
