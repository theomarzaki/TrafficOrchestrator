#include <utility>

#include <utility>

#include <utility>

#include <utility>

#include <utility>

#include <utility>

#include <utility>

// This script is a class for the manuever feedback parsing the information from the v2x Gateway

// Created by: KCL

// Modified by: Omar Nassef(KCL)
#ifndef TO_MANEUVER_FEEDBACK_H
#define TO_MANEUVER_FEEDBACK_H

#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

using namespace std;

class ManeuverFeedback {
private:

	string type;
	string context;
	string origin;
	string version = "1.1.0";
	uint64_t timestamp;
	string uuid_vehicle;
	string uuid_to;
	string uuid_maneuver;
	uint64_t timestamp_message;
	string feedback;
	string reason;
	string signature;

public:

	ManeuverFeedback(string type, string context, string origin, string version, uint64_t timestamp,
	string uuid_vehicle, string uuid_to, string uuid_maneuver, uint64_t timestamp_message, string feedback, string reason, string signature):
	type(type),
	context(context),
	origin(origin),
	version(std::move(version)),
	timestamp(timestamp),
	uuid_vehicle(std::move(uuid_vehicle)),
	uuid_to(std::move(uuid_to)),
	uuid_maneuver(std::move(uuid_maneuver)),
	timestamp_message(timestamp_message),
	feedback(std::move(feedback)),
	reason(std::move(reason)),
	signature(std::move(signature))
	{
		type = "maneuver_feedback";
		context = "lane_merge";
		origin = "vehicle";
	}

	ManeuverFeedback() {
		type = "maneuver_feedback";
		context = "lane_merge";
		origin = "vehicle";
	}

	string getType();
	string getContext();
	string getOrigin();
	string getVersion();
	uint64_t getTimestamp();
	string getUuidVehicle();
	string getUuidTo();
	string getUuidManeuver();
	uint64_t getTimestampMessage();
	string getFeedback();
	string getReason();
	string getSignature();

	void setType(string);
	void setContext(string);
	void setOrigin(string);
	void setVersion(string);
	void setTimestamp(uint64_t);
	void setUuidVehicle(string);
	void setUuidTo(string);
	void setUuidManeuver(string);
	void setTimestampMessage(uint64_t);
	void setFeedback(string);
	void setReason(string);
	void setSignature(string);


	friend std::ostream & operator<<(std::ostream& os, ManeuverFeedback * maneuverFeed);

	~ManeuverFeedback();

};

#endif