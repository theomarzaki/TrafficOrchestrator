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

#define EARTH_RADIUS 6371000

class Mapper {
private:
    inline static const std::string MONTLHERY_MAP = "handcraft_montlhery_road_path";

    inline static std::shared_ptr<Mapper> mapper;
    inline static std::string pathToMap;

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
        int type;
        int forkFrom;
        int mergeInto;
        int numberOfLanes;
        std::map<int,Lane_Descriptor> lanes{std::map<int,Lane_Descriptor>()};
    };

    std::unique_ptr<std::vector<Road_Descriptor>> roads = std::make_unique<std::vector<Road_Descriptor>>();
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
                roads->push_back(roadStruct);
            }
        } else {
            throw std::runtime_error("[ERROR] File is not a valid JSON");
        }
    }

public:

    enum class Mapper_Result_State{
        ERROR,
        OUT_OF_MAP,
        OUT_OF_ROAD,
        OK,
    };

    struct Gps_Descriptor {
        Mapper_Result_State state;
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
        return value * M_PIf64 / 180.0;
    }

    static double toDegree(double value) {
        return value * 180.0 / M_PIf64;
    }

    static double distanceBetween2GPSCoordinates(double latitude, double longitude,
                                                 double latitude2, double longitude2) {  // in meters not in imperials shits
        latitude = toRadian(latitude);
        longitude = toRadian(longitude);
        latitude2 = toRadian(latitude2);
        longitude2 = toRadian(longitude2);

        auto deltalat {latitude2 - latitude};
        auto deltalong {longitude2 - longitude};

        double a = sin(deltalat/2) * sin(deltalat/2) + cos(latitude) * cos(longitude) * sin(deltalong/2) * sin(deltalong/2);
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

    Gps_Descriptor getPositionDescriptor(double latitude, double longitude) {

        Gps_Descriptor nearestDescription;
        nearestDescription.roadId = -1;
        nearestDescription.laneId = -1;
        nearestDescription.roadName = "";
        nearestDescription.state = Mapper_Result_State::ERROR;

        auto distance = std::numeric_limits<double>::max();
        int index = -1;
        int maxIndex = 0;

        for (auto& road : *roads) { // TODO Change basic optimum search
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
                        index = i;
                    }
                }
            }
        }

        if (index > -1) {
            if (distance < 10000) { // TODO Find the extremum of the serie and define the limit with it.
                auto lane = roads->at(nearestDescription.roadId).lanes.find(nearestDescription.laneId)->second;
                auto node = lane.nodes.at(index);

                Lane_Node* compareNode;

                if (index == 0) {
                    compareNode = &lane.nodes.at(index +1);
                } else if (index < maxIndex) {
                    auto previousNode = lane.nodes.at(index -1);
                    auto nextNode = lane.nodes.at(index +1);

                    auto distanceToPrevious = distanceBetween2GPSCoordinates(previousNode.latitude, previousNode.longitude, latitude, longitude);
                    auto distanceToNext = distanceBetween2GPSCoordinates(nextNode.latitude, nextNode.longitude, latitude, longitude);

                    compareNode = distanceToPrevious < distanceToNext ? &previousNode : &nextNode;
                } else {
                    compareNode = &lane.nodes.at(index -1);
                }

                auto xH = distanceBetween2GPSCoordinates(compareNode->latitude, compareNode->longitude, compareNode->latitude, node.longitude);
                auto yH = distanceBetween2GPSCoordinates(compareNode->latitude, compareNode->longitude, node.latitude, compareNode->longitude);
                auto xP = distanceBetween2GPSCoordinates(compareNode->latitude, compareNode->longitude, compareNode->latitude, longitude);
                auto yP = distanceBetween2GPSCoordinates(compareNode->latitude, compareNode->longitude, latitude, compareNode->longitude);

                xH = node.latitude < compareNode->latitude ? -xH : xH;
                yH = node.longitude < compareNode->longitude ? -yH : yH;

                xP = latitude < compareNode->latitude ? -xP : xP;
                yP = longitude < compareNode->longitude ? -yP : yP;

//                std::cout << "Dist : " << distanceBetweenAPointAndAStraightLine(xP, yP, 0, 0, xH, yH) << std::endl;

                if (distanceBetweenAPointAndAStraightLine(xP, yP, 0, 0, xH, yH) <= lane.size/2) { // TODO implement Square B-Spline distance check
                    nearestDescription.state = Mapper_Result_State::OK;
                }
            } else {
                nearestDescription.state = Mapper_Result_State::OUT_OF_MAP;
            }
        }

        return std::move(nearestDescription);
    }
};

//int main() {
//
//    double vectors[9][2] = {{48.623256, 2.242104},  // lane0 in the grass
//                            {48.623206, 2.242140},  // lane1
//                            {48.623183, 2.242167},  // lane2
//                            {48.623162, 2.241968},  // lane1 on the emergency stop strip
//                            {48.623140, 2.241996},  // lane1
//                            {48.623115, 2.242018},  // lane2
//                            {48.623107, 2.241817},  // lane1 on the emergency stop strip
//                            {48.623083, 2.241847},  // lane1
//                            {48.623055, 2.241863}};  // lane2
//
//    for (auto& vector : vectors) {
//        Mapper::Gps_Descriptor buff = Mapper::getMapper()->getPositionDescriptor(vector[0],vector[1]);
//        std::cout << "Lane : " << buff.laneId << std::endl;
//        std::cout << "Status : " << buff.state << "\n\n"; // 3 = OK , 2 = OUT_OF_ROAD
//    }
//}
