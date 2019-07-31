// this script is a class for the manuever recommendation to parse data to JSON format
//  for the v2X gatway to recieve

// Created by: KCL

// Modified by: Omar Nassef (KCL)

#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include "maneuver_recommendation.h"

std::string ManeuverRecommendation::getType() { return type; }
std::string ManeuverRecommendation::getContext() { return context; }
std::string ManeuverRecommendation::getOrigin() { return origin; }
std::string ManeuverRecommendation::getVersion() { return version; }
uint64_t ManeuverRecommendation::getTimestamp() { return timestamp; }
std::string ManeuverRecommendation::getUuidVehicle() { return uuid_vehicle; }
std::string ManeuverRecommendation::getUuidTo() { return uuid_to; }
std::string ManeuverRecommendation::getUuidManeuver() { return uuid_maneuver; }
std::vector<std::shared_ptr<Waypoint>> ManeuverRecommendation::getWaypoints() { return waypoints; }
uint64_t ManeuverRecommendation::getTimestampAction() { return timestamp_action; }
uint32_t ManeuverRecommendation::getLatitudeAction() { return latitude_action; }
uint32_t ManeuverRecommendation::getLongitudeAction() { return longitude_action; }
uint16_t ManeuverRecommendation::getSpeedAction() { return speed_action; }
uint4 ManeuverRecommendation::getLanePositionAction() { return lane_position; }
std::string ManeuverRecommendation::getSignature() { return signature; }
std::string ManeuverRecommendation::getUUID() { return uuid; }
std::string ManeuverRecommendation::getSourceUUID() { return source_uuid; }
std::string ManeuverRecommendation::getMessageID() { return message_id; }

void ManeuverRecommendation::setType(std::string parameter) { type = std::move(parameter); }
void ManeuverRecommendation::setContext(std::string parameter) { context = std::move(parameter); }
void ManeuverRecommendation::setOrigin(std::string parameter) { origin = std::move(parameter); }
void ManeuverRecommendation::setVersion(std::string parameter) { version = std::move(parameter); }
void ManeuverRecommendation::setTimestamp(uint64_t parameter) { timestamp = parameter; }
void ManeuverRecommendation::setUuidVehicle(std::string parameter) { uuid_vehicle = std::move(parameter); }
void ManeuverRecommendation::setUuidTo(std::string parameter) { uuid_to = std::move(parameter); }
void ManeuverRecommendation::setUuidManeuver(std::string parameter) { uuid_maneuver = std::move(parameter); }
void ManeuverRecommendation::setWaypoints(std::vector<std::shared_ptr<Waypoint>> waypointVector) { waypoints = std::move(waypointVector); }
void ManeuverRecommendation::setTimestampAction(uint64_t parameter) { timestamp_action = parameter; }
void ManeuverRecommendation::setLatitudeAction(uint32_t parameter) { latitude_action = parameter; }
void ManeuverRecommendation::setLongitudeAction(uint32_t parameter) { longitude_action = parameter; }
void ManeuverRecommendation::setSpeedAction(uint16_t parameter) { speed_action = parameter; }
void ManeuverRecommendation::setLanePositionAction(uint4 parameter) { lane_position = parameter; }
void ManeuverRecommendation::setSignature(std::string parameter) { signature = std::move(parameter); }
void ManeuverRecommendation::setUUID(std::string parameter) { uuid = std::move(parameter); }
void ManeuverRecommendation::setSourceUUID(std::string parameter) { source_uuid = std::move(parameter); }
void ManeuverRecommendation::setMessageID(std::string parameter) { message_id = std::move(parameter); }

void ManeuverRecommendation::addWaypoint(const std::shared_ptr<Waypoint> &waypoint) { waypoints.push_back(waypoint); }
void ManeuverRecommendation::emptyWaypoints() { waypoints.clear(); }

std::ostream &operator<<(std::ostream &os, ManeuverRecommendation *maneuverRec) {

    os
            << "["
            << maneuverRec->getType()
            << ","
            << maneuverRec->getContext()
            << ","
            << maneuverRec->getOrigin()
            << ","
            << maneuverRec->getVersion()
            << ","
            << maneuverRec->getTimestamp()
            << ","
            << maneuverRec->getUuidVehicle()
            << ","
            << maneuverRec->getUuidTo()
            << ","
            << maneuverRec->getMessageID()
            << ","
            << maneuverRec->getUuidManeuver()
            << ","
            << maneuverRec->getTimestampAction()
            << ","
            << maneuverRec->getLatitudeAction()
            << ","
            << maneuverRec->getLongitudeAction()
            << ","
            << maneuverRec->getSpeedAction()
            << ","
            << maneuverRec->getLanePositionAction()
            << "]\n";
    for (const auto &w : maneuverRec->getWaypoints()) {
        os << w;
    }

    return os;

}
