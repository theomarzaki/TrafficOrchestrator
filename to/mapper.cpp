//
// Created by Johan Maurel <johan.maurel@orange.com> on 15/04/19.
// Copyright (c) Orange Labs all rights reserved.
//

#include "rapidjson/document.h"
#include <vector>

#define MONTLHERY_MAP "handcraft_montlhery_road_path"

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

    std::unique_ptr<vector<Road_Descriptor>> roads = make_unique<vector<Road_Descriptor>>();
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
                struct Road_Descriptor roadStruct;
                roadStruct.id = road.HasMember("_id") ? road["_id"].GetInt() : -1;
                roadStruct.numberOfLanes = road.HasMember("numberOfLanes") ? road["numberOfLanes"].GetInt() : 0;
                roadStruct.name = road["name"].GetString();

                for (auto& lane : road["lanes"].GetArray()) {
                    struct Lane_Descriptor laneStruct;
                    laneStruct.id = lane.HasMember("_id") ? lane["_id"].GetInt() : -1;
                    laneStruct.size = lane.HasMember("_id") ? lane["_id"].GetInt() : 0;

                    for (auto& node : lane["points"].GetArray()) {
                        struct Lane_Node nodeStruct;
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

    struct Gps_Descriptor {
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
                mapper = std::shared_ptr<Mapper>(new Mapper()); // private visibily fix
            }
            return mapper;
        } catch (const std::exception& e) {
            printf("[ERROR] Mapper initializer : %s",e.what());
        }
    }

    Gps_Descriptor getPositionDescriptor(double latitude, double longitude) {

    }
};