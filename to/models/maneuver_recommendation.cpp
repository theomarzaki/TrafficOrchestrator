// this script is a class for the manuever recommendation to parse data to JSON format
//  for the v2X gatway to recieve

// Created by: KCL

// Modified by: Omar Nassef (KCL)

#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include "include/maneuver_recommendation.h"

string ManeuverRecommendation::getType() { return type; }

string ManeuverRecommendation::getContext() { return context; }

string ManeuverRecommendation::getOrigin() { return origin; }

string ManeuverRecommendation::getVersion() { return version; }

uint64_t ManeuverRecommendation::getTimestamp() { return timestamp; }

string ManeuverRecommendation::getUuidVehicle() { return uuid_vehicle; }

string ManeuverRecommendation::getUuidTo() { return uuid_to; }

string ManeuverRecommendation::getUuidManeuver() { return uuid_maneuver; }

vector<std::shared_ptr<Waypoint>> ManeuverRecommendation::getWaypoints() { return waypoints; }

uint64_t ManeuverRecommendation::getTimestampAction() { return timestamp_action; }

uint32_t ManeuverRecommendation::getLatitudeAction() { return latitude_action; }

uint32_t ManeuverRecommendation::getLongitudeAction() { return longitude_action; }

uint16_t ManeuverRecommendation::getSpeedAction() { return speed_action; }

uint4 ManeuverRecommendation::getLanePositionAction() { return lane_position; }

string ManeuverRecommendation::getSignature() { return signature; }

string ManeuverRecommendation::getUUID() { return uuid; }

string ManeuverRecommendation::getSourceUUID() { return source_uuid; }

string ManeuverRecommendation::getMessageID() { return message_id; }

void ManeuverRecommendation::setType(string parameter) { type = parameter; }

void ManeuverRecommendation::setContext(string parameter) { context = parameter; }

void ManeuverRecommendation::setOrigin(string parameter) { origin = parameter; }

void ManeuverRecommendation::setVersion(string parameter) { version = parameter; }

void ManeuverRecommendation::setTimestamp(uint64_t parameter) { timestamp = parameter; }

void ManeuverRecommendation::setUuidVehicle(string parameter) { uuid_vehicle = parameter; }

void ManeuverRecommendation::setUuidTo(string parameter) { uuid_to = parameter; }

void ManeuverRecommendation::setUuidManeuver(string parameter) { uuid_maneuver = parameter; }

void ManeuverRecommendation::setWaypoints(vector<std::shared_ptr<Waypoint>> waypointVector) { waypoints = std::move(waypointVector); }

void ManeuverRecommendation::setTimestampAction(uint64_t parameter) { timestamp_action = parameter; }

void ManeuverRecommendation::setLatitudeAction(uint32_t parameter) { latitude_action = parameter; }

void ManeuverRecommendation::setLongitudeAction(uint32_t parameter) { longitude_action = parameter; }

void ManeuverRecommendation::setSpeedAction(uint16_t parameter) { speed_action = parameter; }

void ManeuverRecommendation::setLanePositionAction(uint4 parameter) { lane_position = parameter; }

void ManeuverRecommendation::setSignature(string parameter) { signature = parameter; }

void ManeuverRecommendation::setUUID(string parameter) { uuid = parameter; }

void ManeuverRecommendation::setSourceUUID(string parameter) { source_uuid = parameter; }

void ManeuverRecommendation::setMessageID(string parameter) { message_id = parameter; }

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
