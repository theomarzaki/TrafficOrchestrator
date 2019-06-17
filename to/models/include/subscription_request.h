// This script is a class that represents sending a subscription request to the v2x Gateway
// in order to start receiving road users

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#ifndef TO_SUB_REQUEST_H
#define TO_SUB_REQUEST_H

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
        version(version),
		timestamp(timestamp),
		filter(filter),
		shape(shape),
		signature(signature),
		source_uuid(source_uuid),
        request_id(request_id)
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

#endif
