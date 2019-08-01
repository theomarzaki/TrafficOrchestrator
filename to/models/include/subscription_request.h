// This script is a class that represents sending a subscription request to the v2x Gateway
// in order to start receiving road users

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#ifndef TO_SUB_REQUEST_H
#define TO_SUB_REQUEST_H

#include <iostream>
#include <string>
#include <ostream>

class SubscriptionRequest {
	private:
		std::string type;
		std::string context;
		std::string origin;
		std::string version = "1.2.0";
		uint64_t timestamp;
		bool filter;
		std::string shape;
		std::pair <int,int> northeast;
		std::pair <int,int> southwest;
		std::string signature;
		std::string source_uuid;
		std::string destination_uuid;
		int request_id;
		std::string message_id;

	public:
		SubscriptionRequest(std::string type, std::string context,std::string origin, std::string version, uint64_t timestamp,
		bool filter,int request_id,std::string shape, std::string signature,std::string source_uuid) :
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

	friend std::ostream& operator<< (std::ostream& os, SubscriptionRequest * subscriptionReq);

	std::string getType();
	std::string getContext();
	std::string getOrigin();
	std::string getVersion();
	uint64_t getTimestamp();
	bool getFilter();
	std::string getShape();
	std::pair<int,int> getNorthEast();
	std::pair<int,int> getSouthWest();

	std::string getSignature();
	int getRequestId();
	std::string getSourceUUID();
	std::string getDestinationUUID();
	std::string getMessageID();


	void setType(std::string);
	void setContext(std::string);
	void setOrigin(std::string);
	void setVersion(std::string);
	void setTimestamp(uint64_t);
	void setFilter(bool);
	void setShape(std::string);
	void setLongitude(double);
	void setLatitude(double);
	void setRadius(double);
	void setSignature(std::string);
	void setRequestId(int);
	void setSourceUUID(std::string);
	void setDestinationUUID(std::string);
	void setMessageID(std::string);
	void setNorthEast(std::pair<int,int>);
	void setSouthWest(std::pair<int,int>);

};

#endif
