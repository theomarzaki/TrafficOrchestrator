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
		pair <int,int> northeast;
		pair <int,int> southwest;
		string signature;
		string source_uuid;
		string destination_uuid;
		int request_id;
		string message_id;

	public:
		SubscriptionRequest(string type, string context,string origin, string version, uint64_t timestamp,
		bool filter,int request_id,string shape, string signature,string source_uuid) :
		type(type),
		context(context),
		origin(origin),
		timestamp(timestamp),
		filter(filter),
		shape(shape),
		signature(signature),
		request_id(request_id),
		source_uuid(source_uuid)
		{
			type = "subscription_request";
			context = "subscriptions";
			origin = "traffic_orchestrator";
			source_uuid = "traffic_orchestrator";
			destination_uuid = "v2x_gateway";
		}

		SubscriptionRequest() {
			type = "subscription_request";
			context = "subscriptions";
			origin = "traffic_orchestrator";
			source_uuid = "traffic_orchestrator";
			destination_uuid = "v2x_gateway";
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
	pair<int,int> getNorthEast();
	pair<int,int> getSouthWest();

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
	void setNorthEast(pair<int,int>);
	void setSouthWest(pair<int,int>);

};

SubscriptionRequest::~SubscriptionRequest() {}

string SubscriptionRequest::getType(){return type;}
string SubscriptionRequest::getContext(){return context;}
string SubscriptionRequest::getOrigin(){return origin;}
string SubscriptionRequest::getVersion(){return version;}
uint64_t SubscriptionRequest::getTimestamp(){return timestamp;}
bool SubscriptionRequest::getFilter(){return filter;}
string SubscriptionRequest::getShape(){return shape;};
string SubscriptionRequest::getSignature(){return signature;}
int SubscriptionRequest::getRequestId(){return request_id;}
string SubscriptionRequest::getSourceUUID(){return source_uuid;}
string SubscriptionRequest::getDestinationUUID(){return destination_uuid;}
string SubscriptionRequest::getMessageID(){return message_id;}
pair<int,int> SubscriptionRequest::getNorthEast(){return northeast;}
pair<int,int> SubscriptionRequest::getSouthWest(){return southwest;}

void SubscriptionRequest::setType(string parameter){type = parameter;}
void SubscriptionRequest::setContext(string parameter){context = parameter;}
void SubscriptionRequest::setOrigin(string parameter){origin = parameter;}
void SubscriptionRequest::setVersion(string parameter){version = parameter;}
void SubscriptionRequest::setTimestamp(uint64_t parameter){timestamp = parameter;}
void SubscriptionRequest::setFilter(bool parameter){filter = parameter;}
void SubscriptionRequest::setShape(string parameter){shape = parameter;}
void SubscriptionRequest::setSignature(string parameter){signature = parameter;}
void SubscriptionRequest::setRequestId(int parameter){request_id = parameter;}
void SubscriptionRequest::setSourceUUID(string parameter){source_uuid = parameter;}
void SubscriptionRequest::setDestinationUUID(string parameter){destination_uuid = parameter;}
void SubscriptionRequest::setMessageID(string parameter){message_id = parameter;}
void SubscriptionRequest::setNorthEast(pair<int,int> parameter){northeast = make_pair(parameter.first,parameter.second);}
void SubscriptionRequest::setSouthWest(pair<int,int> parameter){southwest = make_pair(parameter.first,parameter.second);}

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
  << subscriptionReq->getSignature()
  << "]\n";
  return os;

}
