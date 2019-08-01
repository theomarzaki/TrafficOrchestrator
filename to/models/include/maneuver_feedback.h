// This script is a class for the manuever feedback parsing the information from the v2x Gateway

// Created by: KCL

// Modified by: Omar Nassef(KCL)
#ifndef TO_MANEUVER_FEEDBACK_H
#define TO_MANEUVER_FEEDBACK_H

#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>

class ManeuverFeedback {
private:

	std::string type{"maneuver_feedback"};
	std::string context{"lane_merge"};
	std::string origin{"vehicle"};
	std::string version{"1.1.0"};
	uint64_t timestamp{0};
	std::string uuid_vehicle{""};
	std::string uuid_to{""};
	std::string uuid_maneuver{""};
	uint64_t timestamp_message{0};
	std::string feedback{""};
	std::string reason{""};
	std::string signature{""};

public:

	ManeuverFeedback(std::string type, std::string context, std::string origin, std::string version, uint64_t timestamp,
	std::string uuid_vehicle, std::string uuid_to, std::string uuid_maneuver, uint64_t timestamp_message, std::string feedback, std::string reason, std::string signature);
    ManeuverFeedback() = default;

    ~ManeuverFeedback() = default;

	std::string getType();
	std::string getContext();
	std::string getOrigin();
	std::string getVersion();
	uint64_t getTimestamp();
	std::string getUuidVehicle();
	std::string getUuidTo();
	std::string getUuidManeuver();
	uint64_t getTimestampMessage();
	std::string getFeedback();
	std::string getReason();
	std::string getSignature();

	void setType(std::string);
	void setContext(std::string);
	void setOrigin(std::string);
	void setVersion(std::string);
	void setTimestamp(uint64_t);
	void setUuidVehicle(std::string);
	void setUuidTo(std::string);
	void setUuidManeuver(std::string);
	void setTimestampMessage(uint64_t);
	void setFeedback(std::string);
	void setReason(std::string);
	void setSignature(std::string);


	friend std::ostream & operator<<(std::ostream& os, ManeuverFeedback * maneuverFeed);

};

#endif