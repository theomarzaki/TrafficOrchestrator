// this script models the unsubscription response sent by the TO presented in class structure

// Created by: KCL

// Modified by: Omar Nassef (KCL)
#ifndef TO_UNSUB_REQUEST_H
#define TO_UNSUB_REQUEST_H

#include <iostream>
#include <string>
#include <ostream>

using namespace std;

class UnsubscriptionRequest {
private:
	string type;
	string context;
	string origin;
	string version = "1.2.0";
	uint64_t timestamp;
	int request_id;
	int subscriptionId;
	string signature;
	string source_uuid;
	string destination_uuid;

public:

	UnsubscriptionRequest(string type, string context, string origin, string version,
	uint64_t timestamp, int subscriptionId,int request_id, string signature, string source_uuid):
	type(type),
	context(context),
	origin(origin),
	version(version),
	timestamp(timestamp),
	subscriptionId(subscriptionId),
	signature(signature),
	request_id(request_id),
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

	friend std::ostream& operator<< (ostream& os, UnsubscriptionRequest * unsubscriptionReq);

	string getType();
	string getContext();
	string getOrigin();
	string getVersion();
	uint64_t getTimestamp();
	int getSubscriptionId();
	string getSignature();
	int getRequestId();
	string getSourceUUID();
	string getDestinationUUID();

	void setType(string);
	void setContext(string);
	void setOrigin(string);
	void setVersion(string);
	void setTimestamp(uint64_t);
	void setSubscriptionId(int);
	void setSignature(string);
	void setRequestId(int);
	void setSourceUUID(string);
	void setDestinationUUID(string);

};

#endif