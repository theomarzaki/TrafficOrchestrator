#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

using namespace std;
using std::cout;
using std::string;
using std::ostream;
using std::tuple;

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
	version(version),
	timestamp(timestamp),
	uuid_vehicle(uuid_vehicle),
	uuid_to(uuid_to),
	uuid_maneuver(uuid_maneuver),
	timestamp_message(timestamp_message),
	feedback(feedback),
	reason(reason),
	signature(signature)
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
