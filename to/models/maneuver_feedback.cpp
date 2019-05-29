// This script is a class for the manuever feedback parsing the information from the v2x Gateway

// Created by: KCL

// Modified by: Omar Nassef(KCL)


#include "include/maneuver_feedback.h"

ManeuverFeedback::~ManeuverFeedback() {}

string ManeuverFeedback::getType(){return type;}
string ManeuverFeedback::getContext(){return context;}
string ManeuverFeedback::getOrigin(){return origin;}
string ManeuverFeedback::getVersion(){return version;}
uint64_t ManeuverFeedback::getTimestamp(){return timestamp;}
string ManeuverFeedback::getUuidVehicle(){return uuid_vehicle;}
string ManeuverFeedback::getUuidTo(){return uuid_to;}
string ManeuverFeedback::getUuidManeuver(){return uuid_maneuver;}
uint64_t ManeuverFeedback::getTimestampMessage(){return timestamp_message;}
string ManeuverFeedback::getFeedback(){return feedback;}
string ManeuverFeedback::getReason(){return reason;}
string ManeuverFeedback::getSignature(){return signature;}

void ManeuverFeedback::setType(string parameter){type = parameter;}
void ManeuverFeedback::setContext(string parameter){context = parameter;}
void ManeuverFeedback::setOrigin(string parameter){origin = parameter;}
void ManeuverFeedback::setVersion(string parameter){version = parameter;}
void ManeuverFeedback::setTimestamp(uint64_t parameter){timestamp = parameter;}
void ManeuverFeedback::setUuidVehicle(string parameter){uuid_vehicle = parameter;}
void ManeuverFeedback::setUuidTo(string parameter){uuid_to = parameter;}
void ManeuverFeedback::setUuidManeuver(string parameter){uuid_maneuver = parameter;}
void ManeuverFeedback::setTimestampMessage(uint64_t parameter){timestamp_message = parameter;}
void ManeuverFeedback::setFeedback(string parameter){feedback = parameter;}
void ManeuverFeedback::setReason(string parameter) {reason = parameter;}
void ManeuverFeedback::setSignature(string parameter){signature = parameter;}

std::ostream& operator<<(std::ostream& os, ManeuverFeedback * maneuverFeed) {

	os
	<< "["
	<< maneuverFeed->getType()
	<< ","
	<< maneuverFeed->getContext()
	<< ","
	<< maneuverFeed->getOrigin()
	<< ","
	<< maneuverFeed->getVersion()
	<< ","
	<< maneuverFeed->getTimestamp()
	<< ","
	<< maneuverFeed->getUuidVehicle()
	<< ","
	<< maneuverFeed->getUuidTo()
	<< ","
	<< maneuverFeed->getUuidManeuver()
	<< ","
	<< maneuverFeed->getTimestampMessage()
	<< ","
	<< maneuverFeed->getFeedback()
	<< ","
	<< maneuverFeed->getReason()
	<< ","
	<< maneuverFeed->getSignature()
	<< "]\n";

	return os;
}
