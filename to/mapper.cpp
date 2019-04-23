//
// Created by Johan Maurel <johan.maurel@orange.com> on 15/04/19.
// Copyright (c) Orange Labs all rights reserved.
//

#include "rapidjson/document.h"
#include <vector>
#include <limits>
#include <math.h>
#include <array>

#define MONTLHERY_MAP "handcraft_montlhery_road_path"
#define EARTH_RADIUS 6371000

class Mapper {
private:
    static std::shared_ptr<Mapper> mapper;
    static std::string pathToMap;

    struct Lane_Node {
        double latitude;
        double longitude;
    };

    struct Lane_Descriptor {
        int id;
        float size;
        vector<Lane_Node> nodes{vector<Lane_Node>()};
    };

    struct Road_Descriptor {
        int id;
        std::string name;
        int type;
        int forkFrom;
        int mergeInto;
        int numberOfLanes;
        vector<Lane_Descriptor> lanes{vector<Lane_Descriptor>()};
    };

    std::unique_ptr<vector<Road_Descriptor>> roads = std::make_unique<vector<Road_Descriptor>>();
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

        auto json = std::make_unique<Document>();
        if (!jsonString.empty()) {
            json->Parse(jsonString.c_str());
            createMapContainer(std::move(json));
        } else {
            throw std::runtime_error("[ERROR] File is empty");
        }
    }

    void createMapContainer(std::unique_ptr<Document> json) {
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
                    laneStruct.size = lane.HasMember("_id") ? lane["_id"].GetInt() : 0;

                    for (auto& node : lane["points"].GetArray()) {
                        Lane_Node nodeStruct;
                        nodeStruct.latitude = node.HasMember("latitude") ? node["latitude"].GetDouble() : 120.0; // Aka impossible
                        nodeStruct.longitude = node.HasMember("longitude") ? node["longitude"].GetDouble() : 220.0;
                        laneStruct.nodes.push_back(nodeStruct);
                    }
                    roadStruct.lanes.push_back(laneStruct);
                }
                roads->push_back(roadStruct);
            }
        } else {
            throw std::runtime_error("[ERROR] File is not a valid JSON");
        }
    }

public:

    enum Mapper_Result_State{
        OUT_OF_MAP,
        OUT_OF_ROAD,
        ON,
    };

    struct Gps_Descriptor {
        Mapper_Result_State state;
        int laneId;
        int roadId;
        std::string roadName;
    };

    static bool setMap(const std::string& mapPath) {
        struct stat buffer;
        if ( stat(std::string("./include/"+mapPath+".json").c_str(), &buffer) == 0) {
            printf("[INFO] Map path set");
            pathToMap = mapPath;
            return true;
        } else {
            printf("[ERROR] '%s' don't exist in the include folder",mapPath.c_str());
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
            printf("[ERROR] Mapper initializer : %s",e.what());
        }
    }

    static double toRadian(double value) {
        return value * M_1_PIf64 / 180.0;
    }

    static double toDegree(double value) {
        return value * 180.0 / M_1_PIf64;
    }

    static double distanceBetween2GPSCoordinates(double latitude, double longitude, double latitude2, double longitude2) {  // in meters not in imperials shits
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

    Gps_Descriptor getPositionDescriptor(double latitude, double longitude) {

        Gps_Descriptor nearestDescription;
        nearestDescription.roadId = -1;
        nearestDescription.laneId = -1;
        nearestDescription.roadName = "";
        nearestDescription.state = Mapper_Result_State::OUT_OF_MAP;

        auto distance = numeric_limits<double>::max();

        for (auto& road : *roads) { // TODO Change basic optimum search
            for (auto& lane : road.lanes) {
                for (int i=0; i < size(lane.nodes); i++) {
                    auto nodeDistance = distanceBetween2GPSCoordinates(lane.nodes.at(i).latitude, lane.nodes.at(i).longitude, latitude, longitude);
                    if (nodeDistance < distance) {
                        distance = nodeDistance;
                        nearestDescription.roadId = road.id;
                        nearestDescription.laneId = lane.id;
                        nearestDescription.state = Mapper_Result_State::ON;
                        nearestDescription.roadName = road.name;
                    }
                }
            }
        }
        return std::move(nearestDescription);
    }
};