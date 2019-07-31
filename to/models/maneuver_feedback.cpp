// This script is a class for the manuever feedback parsing the information from the v2x Gateway

// Created by: KCL

// Modified by: Omar Nassef(KCL)

#include "maneuver_feedback.h"

ManeuverFeedback::ManeuverFeedback(std::string type, std::string context, std::string origin, std::string version,
                                   uint64_t timestamp, std::string uuid_vehicle, std::string uuid_to,
                                   std::string uuid_maneuver, uint64_t timestamp_message, std::string feedback,
                                   std::string reason, std::string signature) :
    type(std::move(type)),
    context(std::move(context)),
    origin(std::move(origin)),
    version(std::move(version)),
    timestamp(timestamp),
    uuid_vehicle(std::move(uuid_vehicle)),
    uuid_to(std::move(uuid_to)),
    uuid_maneuver(std::move(uuid_maneuver)),
    timestamp_message(timestamp_message),
    feedback(std::move(feedback)),
    reason(std::move(reason)),
    signature(std::move(signature))
{}

std::string ManeuverFeedback::getType(){return type;}
std::string ManeuverFeedback::getContext(){return context;}
std::string ManeuverFeedback::getOrigin(){return origin;}
std::string ManeuverFeedback::getVersion(){return version;}
uint64_t ManeuverFeedback::getTimestamp(){return timestamp;}
std::string ManeuverFeedback::getUuidVehicle(){return uuid_vehicle;}
std::string ManeuverFeedback::getUuidTo(){return uuid_to;}
std::string ManeuverFeedback::getUuidManeuver(){return uuid_maneuver;}
uint64_t ManeuverFeedback::getTimestampMessage(){return timestamp_message;}
std::string ManeuverFeedback::getFeedback(){return feedback;}
std::string ManeuverFeedback::getReason(){return reason;}
std::string ManeuverFeedback::getSignature(){return signature;}

void ManeuverFeedback::setType(std::string parameter){type = std::move(parameter);}
void ManeuverFeedback::setContext(std::string parameter){context = std::move(parameter);}
void ManeuverFeedback::setOrigin(std::string parameter){origin = std::move(parameter);}
void ManeuverFeedback::setVersion(std::string parameter){version = std::move(parameter);}
void ManeuverFeedback::setTimestamp(uint64_t parameter){timestamp = parameter;}
void ManeuverFeedback::setUuidVehicle(std::string parameter){uuid_vehicle = std::move(parameter);}
void ManeuverFeedback::setUuidTo(std::string parameter){uuid_to = std::move(parameter);}
void ManeuverFeedback::setUuidManeuver(std::string parameter){uuid_maneuver = std::move(parameter);}
void ManeuverFeedback::setTimestampMessage(uint64_t parameter){timestamp_message = parameter;}
void ManeuverFeedback::setFeedback(std::string parameter){feedback = std::move(parameter);}
void ManeuverFeedback::setReason(std::string parameter) {reason = std::move(parameter);}
void ManeuverFeedback::setSignature(std::string parameter){signature = std::move(parameter);}

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
