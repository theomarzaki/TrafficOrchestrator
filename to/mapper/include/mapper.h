//
// Created by Johan Maurel <johan.maurel@orange.com> on 15/04/19.
// Copyright (c) Orange Labs all rights reserved.
//
#ifndef TO_MAPPER_H
#define TO_MAPPER_H

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

#include "gpstools.h"
#include "logger.h"

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

    Mapper();

    void createMapContainer(std::unique_ptr<rapidjson::Document> json);

public:

    enum Mapper_Result_State{
        ERROR,
        OUT_OF_MAP,
        OUT_OF_ROAD,
        OK,
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
        double heading;
        std::string roadName;
    };

    static bool setMap(const std::string& mapPath);

    static std::shared_ptr<Mapper> getMapper();

    static double toRadian(double value);
    static double toDegree(double value);

    static double getCoeficient(   double xP, double yP,
                                    double xH, double yH );

    static double distanceBetween2GPSCoordinates(double latitude, double longitude,
                                                 double latitude2, double longitude2);

    static double distanceBetweenAPointAndAStraightLine(double xP, double yP,
                                                        double xL, double yL,
                                                        double xH, double yH);

    static double getSpeed(  double speed,
                              double acceleration,
                              double time );

    static double getDistance(  double speed,
                               double accelleration,
                               double time);

    static Gps_Point projectGpsPoint(  Gps_Point coord,
                                        double distance,
                                        double heading);

    Gps_Descriptor getPositionDescriptor(double latitude, double longitude, int forcedRoadID = -1, int forcedLaneID = -1);

    std::optional<Merging_Scenario> getFakeCarMergingScenario(double latitude, double longitude, int laneId = -1);

    
};

#endif