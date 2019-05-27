//
// Created by Johan Maurel <johan.maurel@orange.com> on 15/04/19.
// Copyright (c) Orange Labs all rights reserved.
//

#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#include <vector>
#include <limits>
#include <math.h>
#include <array>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <map>
#include <iomanip>

#define EARTH_RADIUS 6371000
#define OUT_OF_MAP_VALUE 10000

class Mapper {
private:
    inline static const std::string MONTLHERY_MAP = "handcraft_montlhery_road_path";

    inline static std::shared_ptr<Mapper> mapper;
    inline static std::string pathToMap;

    enum class Road_Type {
        STRIPE,
        LOOP
    };

    struct Lane_Node {
        double latitude;
        double longitude;
    };

    struct Lane_Descriptor {
        int id;
        float size;
        std::vector<Lane_Node> nodes{std::vector<Lane_Node>()};
    };

    struct Road_Descriptor {
        int id;
        std::string name;
        Road_Type type;
        int forkFrom;
        int mergeInto;
        int numberOfLanes;
        std::map<int,Lane_Descriptor> lanes{std::map<int,Lane_Descriptor>()};
    };

    std::vector<Road_Descriptor> m_roads;

    int numberOfRoad = 0;

    Mapper() {
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

    void createMapContainer(std::unique_ptr<rapidjson::Document> json) { // TODO Change array topography
        if (json->IsObject()) {
            numberOfRoad = json->HasMember("numberOfRoad") ? json->FindMember("numberOfRoad")->value.GetInt() : 0;
            for (auto& road : json->FindMember("roads")->value.GetArray()) {
                Road_Descriptor roadStruct;
                roadStruct.id = road.HasMember("_id") ? road["_id"].GetInt() : -1;
                roadStruct.numberOfLanes = road.HasMember("numberOfLanes") ? road["numberOfLanes"].GetInt() : 0;
                roadStruct.name = road["name"].GetString();
                roadStruct.type = road.HasMember("type") and std::string(road["type"].GetString()) == "loop" ? Road_Type::LOOP : Road_Type::STRIPE;

                for (auto& lane : road["lanes"].GetArray()) {
                    Lane_Descriptor laneStruct;
                    laneStruct.id = lane.HasMember("_id") ? lane["_id"].GetInt() : -1;
                    laneStruct.size = lane.HasMember("size") ? lane["size"].GetFloat() : 0;

                    for (auto& node : lane["points"].GetArray()) {
                        Lane_Node nodeStruct;
                        nodeStruct.latitude = node.HasMember("lat") ? node["lat"].GetDouble() : 120.0; // Aka impossible
                        nodeStruct.longitude = node.HasMember("long") ? node["long"].GetDouble() : 220.0;
                        laneStruct.nodes.push_back(nodeStruct);
                    }
                    roadStruct.lanes.insert( std::pair<int,Lane_Descriptor>(laneStruct.id,laneStruct));
                }
                m_roads.push_back(roadStruct);
            }
        } else {
            throw std::runtime_error("[ERROR] File is not a valid JSON");
        }
    }

public:

    enum Mapper_Result_State{
        ERROR,
        OUT_OF_MAP,
        OUT_OF_ROAD,
        OK,
    };

    struct Gps_Point {
        double latitude;
        double longitude;
    };

    struct Merging_Scenario {
        Gps_Point preceeding;
        Gps_Point following;
    };

    struct Gps_Descriptor {
        Mapper_Result_State state;
        int nodeId;
        int laneId;
        int roadId;
        std::string roadName;
    };

    static bool setMap(const std::string& mapPath) {
        if ( access( std::string("./include/"+mapPath+".json").c_str(), F_OK ) > -1) {
            std::cout << "[INFO] Map path set" << std::endl;
            pathToMap = mapPath;
            return true;
        } else {
            std::cout << "[ERROR] '" << mapPath.c_str() << "' don't exist in the include folder" << std::endl;
            return false;
        }
    }

    static std::shared_ptr<Mapper> getMapper(){
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

    static double toRadian(double value) {
        return value * M_PI / 180.0;
    }

    static double toDegree(double value) {
        return value * 180.0 / M_PI;
    }

    static double distanceBetween2GPSCoordinates(double latitude, double longitude,
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

    static double distanceBetweenAPointAndAStraightLine(double xP, double yP,
                                                        double xL, double yL,
                                                        double xH, double yH) {
        auto coef = (yH - yL) / (xH - xL);
        auto offset = coef*xL - yL;
        return fabs(- coef*xP + yP - offset) / sqrt( pow(coef,2) + 1 );
    }

    Gps_Descriptor getPositionDescriptor(double latitude, double longitude, int forcedRoadID = -1) {

        Gps_Descriptor nearestDescription;
        nearestDescription.roadId = -1;
        nearestDescription.laneId = -1;
        nearestDescription.nodeId = -1;
        nearestDescription.roadName = "";
        nearestDescription.state = Mapper_Result_State::ERROR;

        auto distance = std::numeric_limits<double>::max();
        int maxIndex = 0;

        auto listRoads = m_roads;
        if (forcedRoadID > -1) {
            auto buff = std::vector<Road_Descriptor>();
            buff.push_back(m_roads.at(forcedRoadID));
            listRoads = buff;
        }

        for (auto& road : listRoads) { // TODO Change basic optimum search
            for (auto& lane : road.lanes) {
                for (int i=0; i < size(lane.second.nodes); i++) { // TODO Find match with sector vectors
                    double nodeDistance = distanceBetween2GPSCoordinates(lane.second.nodes.at(i).latitude, lane.second.nodes.at(i).longitude, latitude, longitude);
                    if (nodeDistance < distance) {
                        distance = nodeDistance;
                        nearestDescription.roadId = road.id;
                        nearestDescription.laneId = lane.second.id;
                        nearestDescription.state = Mapper_Result_State::OUT_OF_ROAD;
                        nearestDescription.roadName = road.name;
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

                Lane_Node* compareNode;

                if ( this->m_roads.at(nearestDescription.roadId).type == Road_Type::LOOP ) {

                    Lane_Node* previousNode ;
                    Lane_Node* nextNode ;

                    if (nearestDescription.nodeId == 0) {
                        previousNode = &lane.nodes.at(maxIndex);
                        nextNode = &lane.nodes.at(nearestDescription.nodeId +1);
                    } else if (nearestDescription.nodeId == maxIndex) {
                        previousNode = &lane.nodes.at(nearestDescription.nodeId -1);
                        nextNode = &lane.nodes.at(0);
                    } else {
                        previousNode = &lane.nodes.at(nearestDescription.nodeId -1);
                        nextNode = &lane.nodes.at(nearestDescription.nodeId +1);
                    }

                    auto distanceToPrevious = distanceBetween2GPSCoordinates(previousNode->latitude, previousNode->longitude, latitude, longitude);
                    auto distanceToNext = distanceBetween2GPSCoordinates(nextNode->latitude, nextNode->longitude, latitude, longitude);

                    compareNode = distanceToPrevious < distanceToNext ? previousNode : nextNode;
                } else {
                    if (nearestDescription.nodeId == 0) {
                        compareNode = &lane.nodes.at(nearestDescription.nodeId +1);
                    } else if (nearestDescription.nodeId < maxIndex) {
                        auto previousNode = lane.nodes.at(nearestDescription.nodeId -1);
                        auto nextNode = lane.nodes.at(nearestDescription.nodeId +1);

                        auto distanceToPrevious = distanceBetween2GPSCoordinates(previousNode.latitude, previousNode.longitude, latitude, longitude);
                        auto distanceToNext = distanceBetween2GPSCoordinates(nextNode.latitude, nextNode.longitude, latitude, longitude);

                        compareNode = distanceToPrevious < distanceToNext ? &previousNode : &nextNode;
                    } else {
                        compareNode = &lane.nodes.at(nearestDescription.nodeId -1);
                    }
                }

                auto xH{distanceBetween2GPSCoordinates(compareNode->latitude, compareNode->longitude, compareNode->latitude, node.longitude)};
                auto yH{distanceBetween2GPSCoordinates(compareNode->latitude, compareNode->longitude, node.latitude, compareNode->longitude)};
                auto xP{distanceBetween2GPSCoordinates(compareNode->latitude, compareNode->longitude, compareNode->latitude, longitude)};
                auto yP{distanceBetween2GPSCoordinates(compareNode->latitude, compareNode->longitude, latitude, compareNode->longitude)};

                xH = node.latitude < compareNode->latitude ? -xH : xH;
                yH = node.longitude < compareNode->longitude ? -yH : yH;

                xP = latitude < compareNode->latitude ? -xP : xP;
                yP = longitude < compareNode->longitude ? -yP : yP;

                if (distanceBetweenAPointAndAStraightLine(xP, yP, 0, 0, xH, yH) <= lane.size/2) { // TODO implement Square B-Spline distance check
                    nearestDescription.state = Mapper_Result_State::OK;
                }
            } else {
                nearestDescription.state = Mapper_Result_State::OUT_OF_MAP;
                logger::write(std::string("[WARN] Coordinates out of map, at least "+std::to_string(OUT_OF_MAP_VALUE)+" m away => ("+std::to_string(latitude)+","+std::to_string(longitude)+")"));
            }
        }

        return std::move(nearestDescription);
    }

    std::optional<Merging_Scenario> getFakeCarMergingScenario(double latitude, double longitude) {  // Beware that method is tweaked for our use case. Such as the the road = 1 and lane = 1.
        Gps_Descriptor gps{getPositionDescriptor(latitude,longitude,1)}; // 1 = highway
        if (gps.state != Mapper_Result_State::OUT_OF_MAP) {
            auto nodes{m_roads.at(gps.roadId).lanes.find(1)->second.nodes}; // 1 = First lane
            long max = nodes.size()-1;
            long spread = nodes.size()/6; // size factor

            unsigned int indexFollowing = gps.nodeId - spread < 0 ? max + (gps.nodeId - spread + 1) : gps.nodeId - spread;
            unsigned int indexPreceeding = gps.nodeId + spread > max ? (gps.nodeId + spread) % max - 1 : gps.nodeId + spread;

            Gps_Point carPreceeding = {
                    nodes.at(indexPreceeding).latitude,
                    nodes.at(indexPreceeding).longitude,
            };

            Gps_Point carFollowing = {
                    nodes.at(indexFollowing).latitude,
                    nodes.at(indexFollowing).longitude,
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
};
