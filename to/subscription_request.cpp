// This script is a class that represents sending a subscription request to the v2x Gateway
// in order to start receiving road users

// Created by: KCL

// Modified by: Omar Nassef (KCL)


#include <iostream>
#include <string>
#include <ostream>


using namespace std;

class SubscriptionRequest {
	private:
		string type;
		string context;
		string origin;
		string version = "1.2.0";
		uint64_t timestamp;
		bool filter;
		string shape;
		double longitude;
		double latitude;
		double radius;
		string signature;
		string source_uuid;
		string destination_uuid;
		int request_id;  // added later TODO
		string message_id;

	public:
		SubscriptionRequest(string type, string context,string origin, string version, uint64_t timestamp,
		bool filter,int request_id,string shape, double longitude, double latitude, double radius, string signature,string source_uuid) :
		type(type),
		context(context),
		origin(origin),
		timestamp(timestamp),
		filter(filter),
		shape(shape),
		longitude(longitude),
		latitude(latitude),
		radius(radius),
		signature(signature),
		request_id(request_id), //TODO CHANGED
		source_uuid(source_uuid)
		{
			type = "subscription_request";
			context = "subscriptions"; //changed from subscription_mechanism
			origin = "traffic_orchestrator";
			source_uuid = "traffic_orchestrator";
			destination_uuid = "v2xgw";
		}

		SubscriptionRequest() {
			type = "subscription_request";
			context = "subscriptions"; // same as above
			origin = "traffic_orchestrator";
			source_uuid = "traffic_orchestrator"; // add to json format properly
			destination_uuid = "v2xgw";
		}

	~SubscriptionRequest();

	friend std::ostream& operator<< (ostream& os, SubscriptionRequest * subscriptionReq);

	string getType();
	string getContext();
	string getOrigin();
	string getVersion();
	uint64_t getTimestamp();
	bool getFilter();
	string getShape();
	double getLongitude();
	double getLatitude();
	double getRadius();
	string getSignature();
	int getRequestId();
	string getSourceUUID();
	string getDestinationUUID();
	string getMessageID();


	void setType(string);
	void setContext(string);
	void setOrigin(string);
	void setVersion(string);
	void setTimestamp(uint64_t);
	void setFilter(bool);
	void setShape(string);
	void setLongitude(double);
	void setLatitude(double);
	void setRadius(double);
	void setSignature(string);
	void setRequestId(int);
	void setSourceUUID(string);
	void setDestinationUUID(string);
	void setMessageID(string);

};

SubscriptionRequest::~SubscriptionRequest() {}

string SubscriptionRequest::getType(){return type;}
string SubscriptionRequest::getContext(){return context;}
string SubscriptionRequest::getOrigin(){return origin;}
string SubscriptionRequest::getVersion(){return version;}
uint64_t SubscriptionRequest::getTimestamp(){return timestamp;}
bool SubscriptionRequest::getFilter(){return filter;}
string SubscriptionRequest::getShape(){return shape;};
double SubscriptionRequest::getLongitude(){return longitude;}
double SubscriptionRequest::getLatitude(){return latitude;}
double SubscriptionRequest::getRadius(){return radius;}
string SubscriptionRequest::getSignature(){return signature;}
int SubscriptionRequest::getRequestId(){return request_id;}
string SubscriptionRequest::getSourceUUID(){return source_uuid;}
string SubscriptionRequest::getDestinationUUID(){return destination_uuid;}
string SubscriptionRequest::getMessageID(){return message_id;}

void SubscriptionRequest::setType(string parameter){type = parameter;}
void SubscriptionRequest::setContext(string parameter){context = parameter;}
void SubscriptionRequest::setOrigin(string parameter){origin = parameter;}
void SubscriptionRequest::setVersion(string parameter){version = parameter;}
void SubscriptionRequest::setTimestamp(uint64_t parameter){timestamp = parameter;}
void SubscriptionRequest::setFilter(bool parameter){filter = parameter;}
void SubscriptionRequest::setShape(string parameter){shape = parameter;}
void SubscriptionRequest::setLongitude(double parameter){longitude = parameter;}
void SubscriptionRequest::setLatitude(double parameter){latitude = parameter;}
void SubscriptionRequest::setRadius(double parameter){radius = parameter;}
void SubscriptionRequest::setSignature(string parameter){signature = parameter;}
void SubscriptionRequest::setRequestId(int parameter){request_id = parameter;}
void SubscriptionRequest::setSourceUUID(string parameter){source_uuid = parameter;}
void SubscriptionRequest::setDestinationUUID(string parameter){destination_uuid = parameter;}
void SubscriptionRequest::setMessageID(string parameter){message_id = parameter;}

std::ostream& operator<<(std::ostream& os, SubscriptionRequest * subscriptionReq) {

  os
  << "["
  << subscriptionReq->getType()
  << ","
  << subscriptionReq->getContext()
  << ","
  << subscriptionReq->getOrigin()
  << ","
  << subscriptionReq->getVersion()
  << ","
  << subscriptionReq->getTimestamp()
	<< ","
	<< subscriptionReq->getSourceUUID()
  << ","
	<< subscriptionReq->getDestinationUUID()
	<< ","
  << subscriptionReq->getFilter()
  << ","
	<<subscriptionReq->getRequestId()
	<< ","
  << subscriptionReq->getShape()
  << ","
  << subscriptionReq->getLongitude()
  << ","
  << subscriptionReq->getLatitude()
  << ","
  << subscriptionReq->getRadius()
  << ","
  << subscriptionReq->getSignature()
  << "]\n";
  return os;

}
