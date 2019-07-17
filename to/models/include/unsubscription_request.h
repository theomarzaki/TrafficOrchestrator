// this script models the unsubscription response sent by the TO presented in class structure

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#ifndef TO_UNSUB_REQUEST_H
#define TO_UNSUB_REQUEST_H

#include <iostream>
#include <string>
#include <ostream>

class UnsubscriptionRequest {
private:
	std::string type;
	std::string context;
	std::string origin;
	std::string version = "1.2.0";
	uint64_t timestamp;
	int request_id;
	int subscriptionId;
	std::string signature;
	std::string source_uuid;
	std::string destination_uuid;

public:

	UnsubscriptionRequest(std::string type, std::string context, std::string origin, std::string version,
	uint64_t timestamp, int subscriptionId,int request_id, std::string signature, std::string source_uuid):
	type(type),
	context(context),
	origin(origin),
	version(version),
	timestamp(timestamp),
	request_id(request_id),
    subscriptionId(subscriptionId),
    signature(signature),
	source_uuid(source_uuid)
	{
		type = "unsubscription_request";
		context = "subscriptions";
		origin = "traffic_orchestrator";
		source_uuid = "traffic_orchestrator";
		destination_uuid = "v2x_gateway";
	}

	UnsubscriptionRequest() {
		type = "unsubscription_request";
		context = "subscriptions";
		origin = "traffic_orchestrator";
		source_uuid = "traffic_orchestrator";
		destination_uuid = "v2x_gateway";
	}

	friend std::ostream& operator<< (std::ostream& os, UnsubscriptionRequest * unsubscriptionReq);

	std::string getType();
	std::string getContext();
	std::string getOrigin();
	std::string getVersion();
	uint64_t getTimestamp();
	int getSubscriptionId();
	std::string getSignature();
	int getRequestId();
	std::string getSourceUUID();
	std::string getDestinationUUID();

	void setType(std::string);
	void setContext(std::string);
	void setOrigin(std::string);
	void setVersion(std::string);
	void setTimestamp(uint64_t);
	void setSubscriptionId(int);
	void setSignature(std::string);
	void setRequestId(int);
	void setSourceUUID(std::string);
	void setDestinationUUID(std::string);

};

#endif